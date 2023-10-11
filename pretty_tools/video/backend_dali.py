from ContionTrack.datasets.utils.dali_utils import build_video_pipeline, VideoLoader
from pathlib import Path
import os
from pretty_tools import X_Progress
import torch
import torchvision.transforms as transforms
from pretty_tools.multi_works import BoundThreadPoolExecutor
import queue
import rich
import logging
from .core import num_gpus, Base_VideoConverter

# todo 还没写好适配pretty_tools的接口，以后再改写


class VideoConverter(Base_VideoConverter):
    debug = False
    debug_count = 500

    def __init__(
        self,
        path_input,
        path_output,
        resume=False,
        sequence_length=1,
        num_threads=2,
        gpu_id=0,
    ) -> None:
        # * 初始化，如果当前类没有初始化过 queue_gpu ，就生成一个
        super().__init__(path_input, path_output, resume, num_threads, gpu_id)

        self.sequence_length = sequence_length

        if self.path_out.is_dir():
            self.videos_pip = build_video_pipeline(
                filenames=str(self.path_video),  # * 只能一个一个的实例化
                batch_size=1,
                num_threads=num_threads,
                device_id=self.curr_gpu,
                shuffle=False,
                initial_fill=sequence_length,
                sequence_length=sequence_length,  # * 一次输出的数量
                stride=1,
                step=1,
                pad_last_batch=False,  # 不重复最后一个样本来填充
            )
            self.video_loader = VideoLoader(self.videos_pip)
        else:
            raise FileNotFoundError()

        self.transform = transforms.ToPILImage()

    def check_resume(self):
        pass
        """
        如果是resume，则检查输出路径的文件
        """

    def start_convert(self):
        # todo 这个的resume，还是会从视频中顺序读取图像，无法做到跳帧读取，vpf的读取方式实现了跳帧读取
        self.logger.debug(self.log_head + f"convert start")

        self.queue_gpu[self.curr_gpu].put(id(self))  # * 如果显卡被占用，就阻塞
        # debug部分，只执行100个图像保存就结束
        debug_index = 0
        with BoundThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for imgs, *_ in X_Progress(self.video_loader, f"(cuda:{self.curr_gpu}) process {self.path_out.name}"):
                # 假设你有一个 PyTorch 张量列表 tensors 和对应的文件名列表 filenames
                # 创建一个线程池，将图片保存任务分配给线程池中的多个线程处理
                if len(imgs) != self.sequence_length:
                    self.logger.error(self.log_head + "len not sync")
                filenames = []
                for _ in imgs:
                    filenames.append(str(self.path_out.joinpath(f"{self.curr_frame:06d}.jpg")))  # todo 目前只能用固定的名称来命名输出的文件，之后会使用一个命名生成器来命名
                    self.curr_frame += 1
                for tensor, filename in zip(imgs, filenames):
                    if self.resume and Path(filename).is_file():
                        self.count_resumed += 1
                        continue
                    executor.submit(self.save_image, tensor.permute(2, 0, 1), filename)
                if self.debug:
                    debug_index += 1
                    if debug_index == self.debug_count:
                        break

            if self.resume and self.count_resumed != 0:
                self.logger.info(self.log_head + f"{self.path_out} resumed: {self.count_resumed}")
            executor.shutdown()

    def finish_convert(self):
        self.logger.debug(self.log_head + f"convert over")
        self.queue_gpu[self.curr_gpu].get()

    # 定义一个函数，用于将 PyTorch 张量保存为 JPEG 格式
    def save_image(self, tensor, filename):
        image = self.transform(tensor)
        image.save(filename)
