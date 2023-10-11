from typing import Any, Callable, Generator, Generic, Iterable, Optional, TypeVar
from rich import filesize
from rich.console import Console, Group, RenderableType
from rich.progress import BarColumn, Progress, ProgressColumn, SpinnerColumn, Task, TextColumn, TimeElapsedColumn, TimeRemainingColumn, track
from rich.text import Text


class Pretty_SpeedColumn_item(ProgressColumn):
    """
    用 个数单位 来表达 TransferSpeedColumn，而非数据字节数作为单位
    """

    """Renders human readable transfer speed."""

    def render(self, task: "Task") -> Text:
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("", style="progress.data.speed")
        unit, suffix = filesize.pick_unit_and_suffix(
            int(speed),
            ["", "×10³", "×10⁶", "×10⁹", "×10¹²"],
            1000,
        )
        speed = speed / unit
        return Text(f"{speed:.2f}{suffix} it/s", style="progress.data.speed")


class Pretty_TotalAmountColumn(ProgressColumn):
    """渲染总个数"""

    max_refresh = 0.25

    def render(self, task: "Task") -> Text:
        """Show data completed."""
        total = task.total
        curt = int(task.completed)
        if total != None:
            return Text(f"curt/total ({curt:d}/{total:d}) ", style="progress.filesize.total")
        else:
            return Text(f"count ({curt:d}) ", style="progress.filesize.total")


class Pretty_TimeRemainingColumn(TimeRemainingColumn):
    max_refresh = 0.25
    pass


class Pretty_Text_PercentageColumn(ProgressColumn):
    def render(self, task: "Task") -> Text:
        if task.total is None:
            return Text()
        else:
            return Text(f"{task.percentage:>3.2f}%", style="progress.percentage")
