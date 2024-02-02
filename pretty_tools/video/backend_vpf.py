import queue
import threading
from enum import Enum
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import pycuda.driver as cuda
# * https://www.leiphone.com/category/yanxishe/AUA3C3QTP1lJeBnm.html#
import PyNvCodec as nvc
import torchvision.transforms as transforms
from pretty_tools import X_Progress
from pretty_tools.multi_works import BoundThreadPoolExecutor

from .core import (Backend_Decode_Base, Backend_Encode_Base,
                   Base_VideoConverter, DecodeMethod_noCuda,
                   DecodeMethod_noFFmpeg, DecodeMethod_useCuda,
                   DecodeMethod_useFFmpeg, log_video_batch, num_gpus)

# Initialize CUDA context in main thread
cuda.init()  # type: ignore


class DecodeStatus(Enum):
    # Decoding error.
    DEC_ERR = (0,)
    # Frame was submitted to decoder.
    # No frames are ready for display yet.
    DEC_SUBM = (1,)
    # Frame was submitted to decoder.
    # There's a frame ready for display.
    DEC_READY = 2


class TypeFrame(Enum):
    Type_Numpy = 0
    Type_Cupy = 1
    Type_Torch = 2


class Backend_Decode_VPF(Backend_Decode_Base):
    def __init__(
        self,
        path_video,
        cuda: bool = DecodeMethod_useCuda,
        ffmpeg: bool = DecodeMethod_noFFmpeg,
        num_cache=1,
        num_thread=1,
        data_type=None,
        **kwargs,
    ) -> None:
        import PyNvCodec as nvc

        # 默认使用 gpu 解码
        super().__init__(path_video, cuda, ffmpeg, num_cache=num_cache, num_thread=num_thread)
        if self.cuda is False:
            self.cuda = True
            self.logger.warning("使用 VPF ，这将自动启用 cuda 支持")
        if self.ffmpeg is True:
            self.ffmpeg = False
            self.logger.warning("使用 VPF ，这将取消使用FFmpeg, 因为测试过不太行")

        if "gpu_id" not in kwargs:
            self.gpu_id = 0
            self.logger.warning("没有指明使用的GPU ID, 实例化参数 gpu_id 的缺省值为 0")
        else:
            self.gpu_id = kwargs["gpu_id"]

        if data_type is None:
            self.data_type = TypeFrame.Type_Numpy
            self.logger.warning("没有指明使用的type, 这将返回 numpy 的格式")
        else:
            if data_type == "cupy":
                self.logger.warning("cupy目前没有直接转换的方式，一般通过torch进行转换，这里直接输出torch格式")
                import PytorchNvCodec as pnvc

                self.data_type = TypeFrame.Type_Torch
            if data_type == "numpy":
                self.data_type = TypeFrame.Type_Numpy
            if data_type == "torch":
                import PytorchNvCodec as pnvc

                self.data_type = TypeFrame.Type_Torch

        # * -----------------------  多线程上的设置   ----------------------

        if self.num_thread >= 2:
            raise NotImplementedError("目前暂不支持大于2的线程数")
        self.flag_thread_stop = False  # 当线程停止的时候，就在这里设置为 True

    def setup(self):
        # * ======================上下文流=====================
        # Retain primary CUDA device context and create separate stream per thread.
        device = cuda.Device(self.gpu_id)  # type: ignore
        self.ctx = device.retain_primary_context()  # type: ignore
        self.ctx.push()
        self.str = cuda.Stream()  # type: ignore
        self.ctx.pop()

        self.nv_dec = nvc.PyNvDecoder(str(self.path_video), self.ctx.handle, self.str.handle)

        self.width, self.height = self.nv_dec.Width(), self.nv_dec.Height()
        self.shape[0] = self.width
        self.shape[1] = self.height
        self.shape[2] = 3

        self.num_frames = self.nv_dec.Numframes()
        # * =====================  色彩转换器  ================================
        # * 这里还要配置一个颜色转换器, 转为RGB，注意，色彩转换全部都是处理的Surface数据
        cspace, crange = self.nv_dec.ColorSpace(), self.nv_dec.ColorRange()
        if nvc.ColorSpace.UNSPEC == cspace:
            cspace = nvc.ColorSpace.BT_601
        if nvc.ColorRange.UDEF == crange:
            crange = nvc.ColorRange.MPEG
        self.cc_ctx = nvc.ColorspaceConversionContext(cspace, crange)

        # if self.nv_dec.ColorSpace() != nvc.ColorSpace.BT_709:
        #     self.nvYuv = nvc.PySurfaceConverter(self.width, self.height, self.nv_dec.Format(), nvc.PixelFormat.YUV420, self.ctx.handle, self.str.handle)
        #     self.nvCvt_rgb = nvc.PySurfaceConverter(self.width, self.height, self.nvYuv.Format(), nvc.PixelFormat.RGB, self.ctx.handle, self.str.handle)
        # else:
        #     self.nvCvt_rgb = nvc.PySurfaceConverter(self.width, self.height, self.nv_dec.Format(), nvc.PixelFormat.RGB, self.ctx.handle, self.str.handle)

        # ? -----------------   如果导出的形式是 torch 则还需要进一步转换颜色空间

        self.to_rgb = cconverter(self.width, self.height, cc=self.cc_ctx, handle=(self.ctx.handle, self.str.handle))
        self.to_rgb.add(nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420)
        self.to_rgb.add(nvc.PixelFormat.YUV420, nvc.PixelFormat.RGB)

        if self.data_type == TypeFrame.Type_Torch:
            self.nvCvt_rgb_planar = nvc.PySurfaceConverter(self.width, self.height, nvc.PixelFormat.RGB, nvc.PixelFormat.RGB_PLANAR, self.ctx.handle, self.str.handle)

        # * =====================  Seek 功能 ================================
        self.seek_mode = nvc.SeekMode.PREV_KEY_FRAME
        self.seek_criteria = nvc.SeekCriteria.BY_NUMBER
        self.num_getting = 0  # 这个表明还有多少获取帧的请求是卡在了get这个地方的
        self.event_seek_lock = Event()
        self.event_seek_nolock = Event()
        self.event_seek_nolock.set()  # 表示不锁定

        # * =====================  GPU - CPU 传输 ================================
        self.nvDwn = nvc.PySurfaceDownloader(self.width, self.height, nvc.PixelFormat.RGB, self.ctx.handle, self.str.handle)
        self.nvUpd = nvc.PyFrameUploader(self.width, self.height, nvc.PixelFormat.RGB, self.ctx.handle, self.str.handle)

        # * 启动线程
        self.flag_kill = False
        if self.num_thread > 0:
            self.t = threading.Thread(target=self.threading_get_frame)
            self.t.daemon = True
            self.t.start()

    def terminate(self):
        # vpf没有这种断开数据流的方法，直接断开就行
        self.flag_kill = True
        self.seek_stop()

        pass

    def seek_stop(self):
        # * 调用了seek，先等待上一次的读取请求运行完毕，然后清空缓冲列表中的内容
        # curr_thread_frame 变量会在threading中发生变化，因此先停止线程后，再修改
        self.event_seek_nolock.clear()
        if self.num_getting != 0:
            self.logger.info(f"等待结余请求执行完毕，剩余请求数: {self.num_getting}")
        while self.num_getting != 0:
            import time

            time.sleep(0.001)  # 等待获取图像的请求归零 #! 可能会在这里卡死，因为一直无法等到 num_getting 归零

        self.flag_kill = True
        # 可能会因为在queue中put不进去卡住，所以清空一下queue
        while not self.img_queue.empty():
            # 清空empty
            try:
                self.img_queue.get_nowait()
            except queue.Empty:
                pass

        self.t.join()

    def seek_start(self, index_frame):
        # * ==============   设置初始状态   ===============
        self.img_queue = queue.Queue(self.num_cache)

        self.curr_index_frame = int(index_frame)
        self.next_index_frame = self.curr_index_frame + 1
        self.curr_thread_frame = self.curr_index_frame
        self.next_thread_frame = self.next_index_frame

        # * ==============   恢复   ===============
        self.flag_kill = False
        self.flag_thread_stop = False
        if self.num_thread > 0:
            self.t = threading.Thread(target=self.threading_get_frame)
            self.t.daemon = True
            self.t.start()
        self.event_seek_nolock.set()

    def seek_get_frame(self, index_frame):
        # * 这个类方法是线程安全的
        # * 逐帧提取图像，index_frame是帧索引号，应当从1开始

        #! 调用了 seek方法，就会终止thread，并清空缓冲区的内容，立刻清空多线程缓冲区里的内容，相当于重新开始了

        self.seek_stop()
        # 获取上下文关系
        seek_ctx = nvc.SeekContext(int(index_frame), self.seek_mode, self.seek_criteria)

        rawSurface = self.nv_dec.DecodeSingleSurface(seek_ctx)  #! 这种方式也能正确解码，而且是通过seek模式
        frame = self.threadsafe_last_format_convert(rawSurface)
        self.seek_start(index_frame)
        return frame

    def threadsave_get_frame(self):
        # * 这个应当是线程安全的
        # rawSurface = self.nv_dec.DecodeSingleSurface()
        # frame = self.threadsafe_last_format_convert(rawSurface)
        # return frame

        try:
            rawSurface = self.nv_dec.DecodeSingleSurface()
            frame = self.threadsafe_last_format_convert(rawSurface)
            return frame
        except Exception as e:
            self.logger.error(f"解码出错，帧号为 {self.curr_thread_frame}\n{e.__repr__()}")
            return None

    def threading_get_frame(self):
        # * 不断运行这里，应当只有一个线程运行这个函数
        while self.flag_kill is False and self.curr_thread_frame < self.num_frames:
            # curr_thread_frame 是当前要提取出来的图像的索引号
            # next_thread_frame 是下一个要提取出来的图像的索引号

            self.curr_thread_frame = self.next_thread_frame
            self.next_thread_frame += 1
            frame = self.threadsave_get_frame()
            try:
                if self.flag_kill is False:  # 这个判断放到后面，保证queue不会卡着
                    self.img_queue.put(frame, timeout=1000)
            except Exception as e:
                self.logger.error(e.__repr__())
        self.flag_thread_stop = True  # * 提示拿缓存的线程要终止了

    def get_frame(self):
        # * 如果是多线程，则 get_frame 会运行这里，这个设计是用于被多线程读取已有的缓存的
        self.curr_index_frame = self.next_index_frame
        self.next_index_frame += 1
        self.event_seek_nolock.wait()  #! 锁定seek后，暂停执行get_frame任务，避免又出现queue的请求卡着，并在不合时宜的时候取走生成的数据
        if self.flag_thread_stop:
            self.logger.debug("线程已运行完毕")  # * 解码部分运行完毕后，依然还有几个图没有取走，所以还要继续取

        self.num_getting += 1
        frame = self.img_queue.get()
        self.num_getting -= 1
        return frame

    def threadsafe_last_format_convert(self, rawSurface):
        """
        这个方法是线程安全的，同时调用这个类方法不会引起冲突
        """
        try:
            cvtSurface = self.to_rgb.run(rawSurface)
        except Exception as e:
            self.logger.error(f"{self.curr_thread_frame}/{self.num_frames}\n{e.__repr__()}")

        if self.data_type == TypeFrame.Type_Numpy:
            # * 下载到内存中
            rawFrame = np.ndarray(shape=(cvtSurface.HostSize()), dtype=np.uint8)
            success = self.nvDwn.DownloadSingleSurface(cvtSurface, rawFrame)
            if not success:
                self.logger.warning("Failed to download surface")
            frame = rawFrame.reshape(self.height, self.width, 3)

        if self.data_type == TypeFrame.Type_Torch:
            rgbPSurface = self.nvCvt_rgb_planar.Execute(cvtSurface, self.cc_ctx)
            frame = surface_to_tensor(rgbPSurface)
            frame = frame.reshape(3, self.height, self.width)  # [c, h, w]
        return frame


def surface_to_tensor(surface):
    import PyNvCodec as nvc
    import PytorchNvCodec as pnvc

    if surface.Format() != nvc.PixelFormat.RGB_PLANAR:
        raise RuntimeError("Surface shall be of RGB_PLANAR pixel format")
    surf_plane = surface.PlanePtr()
    img_tensor = pnvc.DptrToTensor(
        surf_plane.GpuMem(),
        surf_plane.Width(),
        surf_plane.Height(),
        surf_plane.Pitch(),
        surf_plane.ElemSize(),
    )
    if img_tensor is None:
        raise RuntimeError("Can not export to tensor.")
    return img_tensor


def tensor_to_surface(img_tensor, gpu_id: int = -1, handle=(None, None)) -> nvc.Surface:
    """
    Converts cuda float tensor to planar rgb surface.
    """
    import PytorchNvCodec as pnvc
    import torch

    if len(img_tensor.shape) != 3 and img_tensor.shape[0] != 3:
        raise RuntimeError("Shape of the tensor must be (3, height, width)")

    tensor_h, tensor_w = img_tensor.shape[1], img_tensor.shape[2]

    img = img_tensor.type(dtype=torch.cuda.ByteTensor)

    if gpu_id >= 0:
        surface = nvc.Surface.Make(nvc.PixelFormat.RGB_PLANAR, tensor_w, tensor_h, gpu_id=gpu_id)
    else:
        surface = nvc.Surface.Make(nvc.PixelFormat.RGB_PLANAR, tensor_w, tensor_h, context=handle[0])

    surf_plane = surface.PlanePtr()
    pnvc.TensorToDptr(
        img,
        surf_plane.GpuMem(),
        surf_plane.Width(),
        surf_plane.Height(),
        surf_plane.Pitch(),
        surf_plane.ElemSize(),
    )

    return surface


class Backend_Encode_VPF(Backend_Encode_Base):
    def __init__(
        self,
        path_video,
        gpu_id,
        width,
        height,
        num_thread=0,
    ) -> None:
        super().__init__(path_video)

        self.gpu_id = gpu_id
        self.width = width
        self.height = height
        self.shape = [height, width, 3]  #! 默认就是3通道图像
        self.num_frames = 0
        self.ctx = cuda.Device(self.gpu_id).retain_primary_context()
        self.ctx.push()
        self.str = cuda.Stream()
        self.ctx.pop()
        self.cc_ctx = nvc.ColorspaceConversionContext(color_space=nvc.ColorSpace.BT_601, color_range=nvc.ColorRange.MPEG)

        # self.nvEnc = nvc.PyNvEncoder(self.dict_Encoder, self.gpu_id)
        self.to_nv12 = cconverter(self.width, self.height, cc=self.cc_ctx, handle=(self.ctx.handle, self.str.handle))
        # self.to_nv12 = cconverter(self.width, self.height, cc=self.cc_ctx, gpu_id=self.gpu_id)
        self.to_nv12.add(nvc.PixelFormat.RGB_PLANAR, nvc.PixelFormat.RGB)
        self.to_nv12.add(nvc.PixelFormat.RGB, nvc.PixelFormat.YUV420)
        self.to_nv12.add(nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12)

        res = str(width) + "x" + str(height)
        self.dict_Encoder = {"preset": "P1", "codec": "h264", "s": res}
        # * =============================  打印信息  ============================================
        self.logger.info(f"输出到 {path_video}")

    def check(self):
        #! 这里存放各种check
        if self.dict_Encoder["codec"] == "h264":
            if self.width > 4096:
                self.logger.warning("H264编码方式中，宽度不能超过4096，自动指定为 hevc(h265)编码方式")
                self.dict_Encoder["codec"] = "hevc"

    def setup(self):
        self.flag_warning = True

        self.check()
        self.nvEnc = nvc.PyNvEncoder(self.dict_Encoder, self.ctx.handle, self.str.handle)
        self.nvUpl = nvc.PyFrameUploader(self.width, self.height, nvc.PixelFormat.YUV420, self.ctx.handle, self.str.handle)

        self.encFrame = np.ndarray(shape=(0), dtype=np.uint8)
        self.framesReceived = 0
        self.framesFlushed = 0

    def __color_convert(self, rawSurface):
        return self.to_nv12.run(rawSurface)

    def encode_frame_builtin(self):
        pass

    def put_frame(self, frame):
        # * width = n * shape[0],  height = m * shape[1]
        # * '(rows cols) c h w -> c (rows h) (cols w) ', rows=n, cols=m
        self.num_frames += 1
        #! 这个的写入是异步的，所以整个流程要经过两次
        # todo 这个函数的功能是能放任何图像，以后补充
        if str(type(frame)) == "<class 'torch.Tensor'>":
            assert frame.shape[0] == 3
            assert frame.shape[1] == self.height
            assert frame.shape[2] == self.width
            rawSurface = tensor_to_surface(frame, handle=(self.ctx.handle, self.str.handle))
            # rawSurface = tensor_to_surface(frame, gpu_id=self.gpu_id)
        if str(type(frame)) == "<class 'numpy.ndarray'>":
            # todo 这个方法还没调试过
            if self.flag_warning:
                self.logger.warning("numpy还没成功调试过，建议先转成torch后再存进来")
            # raise NotImplemented
            rawSurface = self.nvUpl.UploadSingleFrame(frame)

        cvtSurface = self.__color_convert(rawSurface)

        success = self.nvEnc.EncodeSingleSurface(cvtSurface, self.encFrame)

        if success:
            encByteArray = bytearray(self.encFrame)
            self.encFile.write(encByteArray)
            self.framesReceived += 1

    def terminate(self):
        # Encoder is asynchronous, so we need to flush it
        while True:
            success = self.nvEnc.Flush(self.encFrame)
            if success and (self.framesReceived < self.num_frames):
                encByteArray = bytearray(self.encFrame)
                self.encFile.write(encByteArray)
                self.framesReceived += 1
                self.framesFlushed += 1
            else:
                break

        self.logger.info(f"{self.framesReceived} / {self.num_frames} 帧被编码，并写入到文件中.")
        self.encFile.close()


#! 这是一个传入视频文件路径和输出路径后，就能将视频转换为图像文件夹的类
class VideoConverter(Base_VideoConverter):
    data_type = "torch"
    debug = False
    debug_count = 500
    debug_index = 0

    def __init__(
        self,
        path_input,
        path_output,
        resume=False,
        num_threads=4,  # * 目前的vpf只支持线程数为1, 但是这里的num_threads是保存文件时的线程数
        gpu_id=0,
        name="",
    ) -> None:
        super().__init__(path_input, path_output, resume, num_threads, gpu_id)
        self.name = name
        if self.path_out.is_dir():
            self.video = Backend_Decode_VPF(path_input, True, False, num_cache=2 * num_threads, num_thread=1, data_type=self.data_type, gpu_id=gpu_id)
        else:
            raise FileNotFoundError()
        self.video.setup()

        self.num_frames = self.video.num_frames

        self.transform = transforms.ToPILImage()
        self.progress = X_Progress(None, f"(cuda:{self.curr_gpu}) process {self.name}", total_len=self.num_frames)

    def start_convert(self):
        self.logger.debug(self.log_head + f"convert start")
        debug_index = 0
        overpass = False
        with BoundThreadPoolExecutor(max_workers=self.num_threads) as executor:
            #! 绝对不能用progress.a来计数
            while self.index_frame < self.num_frames:
                self.progress.advance()
                # todo 目前只能用固定的名称来命名输出的文件，之后会使用一个命名生成器来命名
                file_name = str(self.path_out.joinpath(f"{self.curr_frame:06d}.jpg"))

                if self.resume and Path(file_name).is_file():
                    self.count_resumed += 1
                    self.index_frame += 1
                    self.curr_frame += 1
                    overpass = True  # triger that jump
                    continue
                elif overpass:
                    overpass = False  # reset
                    img = self.video.seek_get_frame(self.index_frame)  # * index_frame 是帧的索引号，是0开始的，这里已经自加1了，表示下一帧的索引，而不是当前帧的索引
                else:
                    img = self.video.get_frame()

                if self.data_type == "numpy":
                    executor.submit(self.save_image_numpy, img, file_name)
                    # self.save_image_numpy(img, file_name)
                if self.data_type == "torch":
                    executor.submit(self.save_image_torch, img, file_name)

                self.curr_frame += 1
                self.index_frame += 1
            if self.resume and self.count_resumed != 0:
                self.logger.info(self.log_head + f"{self.path_out} resumed: {self.count_resumed}")
            # self.progress.stop()
        executor.shutdown()

    def save_image_numpy(self, img, filename):
        cv2.imwrite(filename, img)

    def save_image_torch(self, img, filename):
        img = self.transform(img)  # * 这里非常耗时
        img.save(filename)

    def finish_convert(self):
        self.video.terminate()


#! -------------------------    以下是官方的API示例   ----------------------------


class cconverter:
    """
    Colorspace conversion chain.
    用 gpu_id 初始化 或者用 handle初始化
    """

    def __init__(self, width: int, height: int, cc, gpu_id: int = -1, handle=(None, None)):
        self.gpu_id = gpu_id
        self.handle = handle

        self.w = width
        self.h = height
        self.chain = []
        self.cc = cc

    def add(self, src_fmt: nvc.PixelFormat, dst_fmt: nvc.PixelFormat) -> None:
        # self.chain.append((nvc.PySurfaceConverter(self.w, self.h, src_fmt, dst_fmt, self.gpu_id), src_fmt))  #* 同时记录进来的种类

        if self.gpu_id >= 0:
            self.chain.append((nvc.PySurfaceConverter(self.w, self.h, src_fmt, dst_fmt, self.gpu_id), src_fmt))  # * 同时记录进来的种类
        else:
            self.chain.append((nvc.PySurfaceConverter(self.w, self.h, src_fmt, dst_fmt, self.handle[0], self.handle[1]), src_fmt))  # * 同时记录进来的种类

    def run(self, src_surface: nvc.Surface) -> nvc.Surface:
        surf = src_surface

        for cvt, fformat in self.chain:
            if surf.Format() != fformat:
                continue
            surf = cvt.Execute(surf, self.cc)
            if surf.Empty():
                raise RuntimeError("Failed to perform color conversion")

        if self.gpu_id >= 0:
            return surf.Clone(self.gpu_id)
        else:
            return surf.Clone(self.handle[0], self.handle[1])
