import argparse
import logging
import threading
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .backend_deffcode import Backend_Decode_Deffcode
from .backend_vpf import Backend_Decode_VPF
from .core import Backend_Decode_Base, Backend_deffcode, Backend_opencv, Backend_vpf, DecodeMethod_noCuda, DecodeMethod_noFFmpeg, DecodeMethod_useCuda, DecodeMethod_useFFmpeg, log_video_batch

parser = argparse.ArgumentParser(description="批量解码")
parser.add_argument("--path_video_dir", type=str, default="/data/Datasets/AIC23_Track1/train/S002")
parser.add_argument("--gpu_id", type=int, default=0, help="-1表示轮流使用所有的GPU")
parser.add_argument("--type", type=str, default="numpy", help="numpy表示返回的是numpy.ndarray类型\ncupy表示返回的是cupy类型\ntorch表示返回的是torch.Tensor类型")

# todo seek模式还没有配置，查了一下，两个后端都是支持seek的

# from memory_profiler import profile


# @profile
class Batch_Video:
    dict_loglevel = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "fatal": logging.FATAL,
    }

    def __init__(
        self,
        list_path: "list[Path | str]",
        backend=Backend_deffcode,
        cuda: bool = DecodeMethod_noCuda,
        ffmpeg: bool = DecodeMethod_noFFmpeg,
        echo_level="warning",
        num_cache=0,
        num_thread=0,
        **kwargs,
    ) -> None:
        self.list_path = [Path(i) for i in list_path]
        self.cuda: bool = cuda  # 默认不使用 gpu 解码
        self.ffmpeg: bool = ffmpeg
        self.num_videos: int = len(self.list_path)
        self.max_mum_frame: int = 0  # 多个视频中，最长的那个视频的帧数
        self.curr_index_frame: int = -1
        self.next_index_frame: int = 0

        self.kwargs = kwargs

        self.backend = backend
        self.Video_Instance = Backend_Decode_Base
        # * ---------------------- 日志登记  -------------------------------
        if echo_level in self.dict_loglevel:
            self.echo_level = self.dict_loglevel[echo_level]
        assert type(self.echo_level) is int
        self.logger = log_video_batch
        self.logger.setLevel(self.echo_level)

        # * ----------------------- 指定后端 ------------------------------
        if self.backend == Backend_deffcode:
            self.Video_Instance = Backend_Decode_Deffcode
        if self.backend == Backend_vpf:
            self.Video_Instance = Backend_Decode_VPF
            if self.cuda is False:
                self.logger.warning("调用VPF后端解码器，自动启用CUDA")
                self.cuda = True
        if self.backend == Backend_opencv:
            raise NotImplemented("OpenCV的后端还没有补充")

        # * ----------------------- 存放数据部分 ------------------------------
        self.list_curr_frames = [None for _ in range(self.num_videos)]
        self.list_next_frames = [None for _ in range(self.num_videos)]
        # * ----------------------- 状态标志 ------------------------------
        self.flag_init = True
        self.flag_close = False
        self.event_ready = threading.Event()
        self.event_empty = threading.Event()  # 这个状态位如果是空的话，就启动轮询
        self.event_kill = threading.Event()

        # * ----------------------- CUDA 检测 -----------------------------

        self.num_avaliable_gpu = 0
        if self.cuda:
            import cupy as cp

            self.gpu_device = []
            while True:
                current_device = cp.cuda.Device(self.num_avaliable_gpu)
                try:
                    self.logger.info(f"检测到显卡 {current_device.pci_bus_id}")
                    self.gpu_device.append(current_device)
                    self.num_avaliable_gpu += 1
                except Exception as e:
                    break
            self.logger.info(f"可用GPU数量: {self.num_avaliable_gpu}")

        self.num_cache = num_cache
        self.setup()

    def setup(self):
        self.event_kill.clear()
        self.event_ready.clear()
        self.event_empty.clear()
        # 把初始化视频的代码放到这里，让整体看起来不那么乱
        self.list_videos: list[Backend_Decode_Base] = []

        #! 这个计算最大帧数的部分一定要放在启动迭代器之前，不然迭代器不知道整个遍历的长度
        for i in range(self.num_videos):
            path_video = self.list_path[i]

            kwargs = deepcopy(self.kwargs)
            if self.backend == Backend_vpf:
                # 如果用的是 vpf，则默认用多个显卡，并且轮回使用
                if "gpu_id" not in self.kwargs:
                    kwargs["gpu_id"] = i % self.num_avaliable_gpu
                    self.logger.info(f"循环调用显卡 {i % self.num_avaliable_gpu}")
                elif self.kwargs["gpu_id"] == -1:
                    kwargs["gpu_id"] = i % self.num_avaliable_gpu
                    self.logger.info(f"循环调用显卡 {i % self.num_avaliable_gpu}")
                else:
                    self.logger.info(f"调用显卡{kwargs['gpu_id'] % self.num_avaliable_gpu}")
                    kwargs["gpu_id"] %= self.num_avaliable_gpu

            video = self.Video_Instance(
                path_video,
                cuda=self.cuda,
                ffmpeg=self.ffmpeg,
                num_cache=self.num_cache,
                **kwargs,
            )
            self.list_videos.append(video)
            video.setup()
            if video.num_frames > self.max_mum_frame:
                self.max_mum_frame = video.num_frames

        self.logger.info(f"Video Batch —— {self.num_videos} 个视频中，最长视频帧数为: {self.max_mum_frame}")

        self.t = threading.Thread(target=self.threading_get_frames)
        self.t.daemon = True
        self.t.start()

    def __next__(self) -> "list[np.ndarray | torch.Tensor | None]":
        self.curr_index_frame = self.next_index_frame
        self.next_index_frame += 1
        if self.curr_index_frame > self.max_mum_frame:  # 已经全部读取完了，这时候关闭掉所有的后端
            self.logger.info("全部视频读取完毕")
            self.event_kill.set()
            self.close()
            raise StopIteration

        self.logger.debug("等待 生产者提供图像 ")
        self.event_ready.wait()  # todo 最后一帧的时候这里会卡死
        self.event_ready.clear()
        # 清空原来的数据

        del self.list_curr_frames
        self.list_curr_frames = self.list_next_frames
        del self.list_next_frames
        self.list_next_frames: "list[np.ndarray | torch.Tensor | None]" = [None for _ in range(self.num_videos)]
        self.event_empty.set()
        return self.list_curr_frames

        # todo 获取frames，简单点，直接用列表形式给出，给出的形式是numpy的或者是cupy的

    def __iter__(self):
        self.curr_index_frame = 0
        self.next_index_frame = 1
        if not self.flag_init:
            self.close()
            self.setup()  # 启动一个新的迭代，首先关闭掉之前的所有
        self.flag_init = False
        return self

    def threading_get_frames(self):
        # 这个是一定会启用的一个线程，不断抓取各个视频生产的图像
        # * 和后端的num_cache、num_thread无关，只要类的状态是 not_filled 就去调用读取函数
        while not self.event_kill.is_set() and self.curr_index_frame < self.max_mum_frame:
            self.event_empty.clear()

            self.logger.debug("等待 event_empty ")
            for i in range(self.num_videos):
                self.list_next_frames[i] = self.list_videos[i].get_frame()  # type: ignore
                assert self.list_next_frames[i] is not None

            self.logger.debug("完成一个批次图像的载入")
            self.event_ready.set()
            self.event_empty.wait()  # 等待当前帧列表被读取

    def close(self):
        if self.flag_close:
            self.logger.info("已经关闭视频流")
            return
        else:
            self.flag_close = True
        self.event_kill.set()
        self.event_empty.set()
        try:
            self.t.join()
            for i in self.list_videos:
                if i is not None:
                    i.terminate()
            self.logger.info("视频流全部终止")
        except Exception as e:
            self.logger.error(e.__repr__())
            self.logger.error("视频流终止失败")

    def __len__(self):
        return self.max_mum_frame


if __name__ == "__main__":
    path_video_dir = Path("/data/Datasets/AIC23_Track1/train/S002")
    args = parser.parse_args()

    list_path_video = []
    for video in path_video_dir.glob("c*"):
        list_path_video.append(video.joinpath("video.mp4"))

    batch_video = Batch_Video(
        list_path_video,
        echo_level="debug",
        cuda=True,
        ffmpeg=False,
        backend=Backend_vpf,
        num_cache=0,
        num_thread=0,
        gpu_id=args.gpu_id,
        data_type=args.type,
    )

    t0_start = time.time()
    t_start = t0_start
    i = 0
    for list_frames in X_Progress(batch_video):
        i += 1
        t_end = time.time()
        print(f"帧率： {1/(t_end-t_start):03f}")
        t_start = time.time()
        # cv2.imwrite("tmp.jpg", list_frames[0])
    t0_end = time.time()
    print(f"总共 读取图像数：{i}")
    print(f"读取完毕 全局耗时: {t0_end-t0_start}")
    batch_video.close()
