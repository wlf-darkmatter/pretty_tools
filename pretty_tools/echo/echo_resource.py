"""
通过调用rich的进度条等功能，实现打印系统资源占用信息
"""
import os
from typing import Optional, Union

from pretty_tools.echo.x_progress import X_Progress
from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.text import Text

optional_item = [
    "username",  # The name of the user that owns the process
    "create_time",  # The process creation time as a floating point number expressed in seconds since the epoch
    "cwd",  # Process current working directory as an absolute path
    "uids",  # Return process UIDs as a (real, effective, saved) #* POSIX
    "gids",  # Return process GIDs as a (real, effective, saved) #* POSIX
    "terminal",  # The terminal associated with this process #* POSIX
    "num_fds",  # Return the number of file descriptors opened by this process #* (POSIX only)
    "num_threads",
    "cpu_percent",
    "cpu_times",
    "memory_info",
    "memory_full_info",
    "memory_percent",
    "open_files",  # Return files opened by process as a list of (path, fd) namedtuples including the absolute file name and file descriptor number.
    "connections",  # Return socket connections opened by process
]

costom_item = ["CPU"]  # todo 之后要补全 CPU, MEM, GPU, DISK, NET 等信息


class Echo_Resource:
    """

    创建实例

    """

    def __init__(self, list_display: Optional[list[str]] = None, live: bool = True) -> None:
        """
        list_display 可选项
        live: True 直接使用当前控制台终端使用的live
        """
        import psutil

        self.pid = os.getpid()
        self.p = psutil.Process(self.pid)

        if list_display is None:
            self.list_display = ["CPU", "MEM"]  # 默认项
        else:
            self.list_display = list_display

        if live is True:
            self.echo = X_Progress._echo_plus
        else:
            self.echo = None

    def __enter__(self):
        if self.echo:
            self.echo.slot_up[id(self)] = self.generate_display  # * 把 可渲染对象放到 slot 里面去
        self.generate_display()
        return self

    def __exit__(self, *exc_info):
        if self.echo:
            del self.echo.slot_up[id(self)]

    def generate_display(self) -> RenderableType:  #! 应当将所有可渲染对象合并成一个
        list_render = []
        for i in self.list_display:
            list_render.append(getattr(self, "_display_" + i)())

        return Group(*list_render)

    #! ================================================================================
    def _display_CPU(self) -> RenderableType:
        import psutil

        cpu_percent = self.p.cpu_percent()
        cpu_idle = psutil.cpu_times_percent().idle
        iowait = psutil.cpu_times_percent().iowait
        return Columns(
            [
                Text.assemble("CPU: ", get_quick_color_ascend(cpu_percent), "%"),
                Text.assemble("idle: ", get_quick_color_descend(cpu_idle), "%"),
                Text.assemble("iowait: ", get_quick_color_ascend(iowait), "%"),
            ],
        )

    def _display_MEM(self) -> RenderableType:
        import psutil

        mem_percent = self.p.memory_percent()
        swap_percent = psutil.swap_memory().percent
        mem_info = self.p.memory_info()
        shm: str = get_quick_pretty(mem_info.shared, True)  # type: ignore
        return Columns(
            [
                Text.assemble("MEM: ", get_quick_color_ascend(mem_percent), "%"),
                Text.assemble("swap: ", get_quick_color_ascend(swap_percent), "%"),
                f"shm: {shm}",
            ]
        )


def get_quick_color_ascend(value: Union[int, float]) -> tuple[str, str]:
    if value < 50:  # * 大于50显示黄色，大于80显示红色
        str_info = (f"{value:0>5.2f}", "green")
    elif value < 80:
        str_info = (f"{value:0>5.2f}", "yellow")
    else:
        str_info = (f"{value:0>5.2f}", "red")
    return str_info


def get_quick_color_descend(value: Union[int, float]) -> tuple[str, str]:
    if value > 50:  # * 大于50显示黄色，大于80显示红色
        str_info = (f"{value:0>5.2f}", "green")
    elif value > 20:
        str_info = (f"{value:0>5.2f}", "yellow")
    else:
        str_info = (f"{value:0>5.2f}", "red")
    return str_info


def get_quick_pretty(v: Union[int, float], pure_str=False):
    v = float(v)
    str_count = "B"
    if v > 1024:
        v /= 1024
        str_count = "KB"
    if v > 1024:
        v /= 1024
        str_count = "MB"
    if v > 1024:
        v /= 1024
        str_count = "GB"
    if v > 1024:
        v /= 1024
        str_count = "TB"
    if pure_str:
        return f"{v:.2f} {str_count}"
    else:
        return v, str_count
