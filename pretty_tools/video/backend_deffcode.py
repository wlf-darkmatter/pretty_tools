import multiprocessing as mp

from multiprocessing import Event
from pathlib import Path

from .core import DecodeMethod_noCuda, DecodeMethod_noFFmpeg, DecodeMethod_useCuda, DecodeMethod_useFFmpeg
from .core import log_video_batch, Backend_Decode_Base, dict_loglevel
from ..echo.x_logging import X_Logging

#! Backend_Deffcode 对应的一种批量解码器，同时解码多个视频，并且同时输出这一批次视频的相同帧
# * https://abhitronix.github.io/deffcode/v0.2.5-stable/
# ? 这里的使用cuda进行加速的方法一定是只有这样吗，目前还不能再细化自定义（因为看不懂文档）
cuda_ffparams = {
    "-vcodec": None,  # use H.264 CUVID Video-decoder
    "-enforce_cv_patch": True,  # enable OpenCV patch for YUV(YUV420p) frames
    "-ffprefixes": [
        "-vsync",
        "0",  # prevent duplicate frames
        "-hwaccel",
        "cuda",  # accelerator
        "-hwaccel_output_format",
        "cuda",  # output accelerator
    ],
}


class Backend_Decode_Deffcode(Backend_Decode_Base):
    def __init__(
        self,
        path_video,
        cuda: bool = DecodeMethod_noCuda,
        ffmpeg: bool = DecodeMethod_useFFmpeg,
        num_cache=0,
        num_thread=0,
        **kwargs,
    ) -> None:
        from deffcode import FFdecoder

        super().__init__(path_video, cuda, ffmpeg, num_cache=num_cache, num_thread=num_thread)
        if self.ffmpeg is False:
            self.ffmpeg = True
            self.logger.warning("使用 Deffcode 后端，这将自动启用 ffmpeg 支持")
        self.cuda = cuda

        if self.num_thread >= 1:
            raise NotImplementedError("目前 Decode_Deffcode 暂不支持多线程")

    def setup(self):
        from deffcode import FFdecoder

        if self.cuda:
            self.decoder = FFdecoder(str(self.path_video), **cuda_ffparams).formulate()

        else:
            self.decoder = FFdecoder(str(self.path_video)).formulate()
        self.generate = self.decoder.generateFrame()

        self.width, self.height = self.decoder._FFdecoder__raw_frame_resolution  # type: ignore
        self.shape[0] = self.width
        self.shape[1] = self.height
        self.shape[2] = 3

        self.num_frames = self.decoder._FFdecoder__raw_frame_num  # type: ignore
        #! 总控制器信号
        self.main_event_kill = Event()
        self.main_event_kill.clear()

    def terminate(self):
        self.main_event_kill.set()
        self.decoder.terminate()

    def get_frame(self):
        self.curr_index_frame = self.next_index_frame
        self.next_index_frame += 1
        if self.num_cache == 0:
            return self.generate.__next__()
