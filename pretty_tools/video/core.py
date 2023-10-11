import logging
import os
import queue
from pathlib import Path

from ..echo.x_logging import X_Logging, build_logging

try:
    from gpuinfo import GPUInfo

    num_gpus = len(GPUInfo.get_info()[1])  # type: ignore
except:
    num_gpus = 0

video_convert_logging = X_Logging("VideoConverter", "log/video_convert", file_log_level=logging.DEBUG)

DecodeMethod_noCuda = False
DecodeMethod_useCuda = True
DecodeMethod_noFFmpeg = False
DecodeMethod_useFFmpeg = True

Backend_opencv = 0  # todo 这个目前没有用上
Backend_deffcode = 1  #! 使用的是 deffcode
Backend_vpf = 2  #! 使用的是 VideoProcessingFramework， 默认启用cuda

log_video_batch = build_logging(logname="video_batch").logger

dict_loglevel = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
}


class Backend_Decode_Base:
    def __init__(
        self,
        path_video: "Path | str",
        cuda: bool = DecodeMethod_noCuda,
        ffmpeg: bool = DecodeMethod_noFFmpeg,
        num_cache=0,  # * 生产消费的存储空间，当num_thread为0的时候该值没有意义
        num_thread=0,  # * 为每个视频提供生产的线程数，如果设为0，则主线程会在这里阻塞，由主线程自己挨个获取图像
    ) -> None:
        # * ========= configuration =========
        self.cuda = cuda
        self.ffmpeg = ffmpeg
        self.num_cache = num_cache
        self.num_thread = num_thread
        # * ========= runtime param =========
        self.curr_index_frame: int = 0
        self.next_index_frame: int = 1
        self.num_frames: int = 0
        self.shape: list[int] = [0, 0, 0]
        self.logger = log_video_batch
        self.verbose = False
        self.curr_thread_frame = 0  # 这个是用于记录在多线程中读取的帧号的索引值
        self.next_thread_frame = 1
        # num_cache和num_thread必须都同时设置
        if num_cache != 0 and num_thread == 0:
            self.num_thread = 1
            self.logger.warning("设定了缓存，则线程默认大于0，此处设置为1")
        if num_thread != 0 and num_cache == 0:
            self.num_cache = 1
            self.logger.warning("设定了线程，则缓存默认大于0，此处设置为1")

        if num_cache > 0:
            self.img_queue = queue.Queue(num_cache)

        # ---------------------------------------------------------
        self.path_video = Path(path_video)
        if not self.path_video.is_file():
            raise FileNotFoundError()

        self.width = -1
        self.height = -1

        # * 多线程状态标志位
        self.flag_kill = False

    def setup(self) -> None:
        raise NotImplemented

    # 这些video实例应当有两种获取帧的方式，一种是多线程的，一种是单线程的，多线程即

    def terminate(self) -> None:
        raise NotImplemented

    def get_frame(self):
        return NotImplementedError()

    def __iter__(self):
        self.setup()
        return self

    def __next__(self):
        return self.get_frame()

    def __len__(self):
        return self.num_frames


class Backend_Encode_Base:
    def __init__(
        self,
        path_video: "Path | str",  # 输出路径
    ) -> None:
        pass
        # * ========= configuration =========

        self.logger = log_video_batch
        self.verbose = False
        self.shape: list[int] = [0, 0, 0]
        # ---------------------------------------------------------
        self.path_video = Path(path_video)
        self.encFile = open(str(path_video), "wb")

    def threading_put_frame(self):
        pass


class Base_VideoConverter:
    logging = video_convert_logging
    logger = logging.logger

    def __init__(
        self,
        path_input,
        path_output,
        resume,
        num_threads,
        gpu_id,
    ) -> None:
        self.progress = None  # * 如果是在子线程中使用，则这个需要在外面指定并在外面结束
        self.resume = resume
        self.curr_gpu = gpu_id
        self.path_video = Path(path_input)
        self.path_out = Path(path_output)
        self.num_threads = num_threads
        assert self.path_video.name.endswith(".mp4")
        if self.path_out.exists() and self.resume is False:
            assert len(os.listdir(self.path_out)) == 0, "需要确保输出目录为空"
        os.makedirs(self.path_out, exist_ok=True)
        self.log_head = f"[GPU:{gpu_id}] | [video:{path_input}]\n"

        self.curr_frame = 1  # * 起始帧应当为1
        self.index_frame = 0  # * 起始帧索引号为0
        self.count_resumed = 0

    def check_resume(self):
        return NotImplemented

    def start_convert(self):
        return NotImplemented

    def finish_convert(self):
        return NotImplemented
