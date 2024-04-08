from typing import Any, Callable, Generator, Generic, Iterable, Optional, TypeVar

from pretty_tools.echo.echo_column.pretty_column import Pretty_SpeedColumn_item, Pretty_Text_PercentageColumn, Pretty_TotalAmountColumn
from rich import filesize
from rich.console import Console, Group, RenderableType
from rich.progress import BarColumn, Progress, ProgressColumn, SpinnerColumn, Task, TextColumn, TimeElapsedColumn, TimeRemainingColumn, track
from rich.text import Text

T = TypeVar("T")
TT = TypeVar("TT")


def init_live():
    """
    rich 设计为 有多个 Console，但是每个Console只能有一个活动的live，
    但是创建一个Progress的时候会强制启动一个自己的live，这时候如果有其他的 进度条或者活动的live在Console中活动着，则会报错，
    所以如果确实存在活动的live，则需要在 Progress调用start()前，改变其内部的live为当前活动的live

    运行逻辑，判断当前全局是否存在一个活动的live
    * 如果是，则新创建的 X_Progress 只能依附于这个live
    * 如果否，则直接使用 X_Progress 自身初始化的live
    """
    from pretty_tools.echo.x_progress import X_Progress
    from rich import get_console

    global_live = get_console()._live
    if global_live is None:
        global_live = X_Progress._echo_plus.live

    return global_live


class Echo_Plus(Progress):
    def init_plus(self):
        """
        拓展初始化，目前留了一个槽位用于显示资源使用情况
        """
        self.slot_up: dict[int, Callable] = {}  # * 对象id号索引的一个 callable 函数，通过调用可以获取更新的值
        self.slot_bottom: dict[int, Callable] = {}
        pass

    def get_renderable(self) -> RenderableType:
        """Get a renderable for the progress display."""
        list_render = []
        if hasattr(self, "slot_up"):
            list_render += [i() for i in self.slot_up.values()]

        list_render += [*self.get_renderables()]
        if hasattr(self, "slot_bottom"):
            list_render += [i() for i in self.slot_bottom.values()]

        renderable = Group(*list_render)
        del list_render
        return renderable


class X_Progress(Generic[T]):
    """
    进度条迭代器


    Args:
        generator (Iterable[T], optional): 迭代对象生成器
        title (str, optional): 进度条标题. Defaults to None.
        total (int, optional): 迭代器总长度. Defaults to None.
            如果 :code:`generator` 不传入，则用 :code:`total` 设置总长度,
            如果传入了 :code:`generator`，则自动使用 :code:`len(generator)` 作为总长度。
        auto_hide (bool, optional): 执行完毕后是否自动隐藏. Defaults to True.

    .. note::

        当 :code:`total` 传入时， 强制使用 :code:`total` 作为总长度


    Warning
    -------
    这个类不支持 **break** 语句

    注意，没有使用with方法进入的进度条，会因为无法处理 **break** 后的进度条假死，如下：

    .. code-block:: python

        for i in X_Progress(range(3), "Main Progress"):
            for j in X_Progress.sub_progress(range(4), f"Minor Progress {i}"):
                for k in X_Progress.sub_progress(range(50), f"Mini Progress {i}-{j}"):
                    time.sleep(0.005)


    **with** 方法 大进度条 by :code:`X_Progress.sub_progress`

    下面这种方法生成的子进度条不在with的管理范围内，如果 **break** 出 **with** 的范围了，内部的所有子进度条 **不会** 被清空

    .. code-block:: python

        with X_Progress(title="Big Progress") as progress:
            for i in range(100):
                time.sleep(0.005)
                progress.advance()
                for j in X_Progress.sub_progress(range(50), "with无关子进度条"):
                    time.sleep(0.001)


    **with** 方法 大进度条 by :code:`progress.sub`

    下面这种方法生成的子进度条是在with管理范围内的，如果 **break** 出 **with** 的范围了，内部的所有子进度条都 **会** 被清空

    .. code-block:: python

        with X_Progress(title="Big Progress") as progress:
            for i in range(100):
                time.sleep(0.005)
                progress.advance()
                for j in progress.sub(range(50), "with相关子进度条"):
                    time.sleep(0.001)


    """

    #! Progress_Plus 修改了 make_tasks_table 部分的封装，使得显示进度条的同时还可以显示其他信息

    _echo_plus = Echo_Plus(
        TextColumn("[progress.description]{task.description}"),
        SpinnerColumn(),
        BarColumn(),
        Pretty_Text_PercentageColumn(),
        Pretty_SpeedColumn_item(),
        TimeElapsedColumn(),
        TextColumn("ETA:"),
        TimeRemainingColumn(),
        Pretty_TotalAmountColumn(),
    )
    _echo_plus.init_plus()

    def __init__(self, generator: Optional[Iterable[T]] = None, title="x_progress", total: Optional[int] = None, auto_hide=False, visible=True) -> None:
        self.task_id = None
        self.task = None
        self.title = title
        self.total = total
        self.auto_hide = auto_hide
        self.parent = None
        self.dict_child = {}
        self.visible = visible
        if generator is not None:
            self.generator = generator
            if total is None and hasattr(generator, "__len__"):
                self.total = len(generator)  # type: ignore
        # * 没有 len 的话，且无法计算获取 len 的话，则不显示进度，只显示速度

    @classmethod
    def start(cls):
        if cls._echo_plus.finished:
            live = init_live()
            cls._echo_plus.live = live  # * 这里第一次运行的时候很可能是自己的live分配给自己
            cls._echo_plus.start()

    @classmethod
    def clean(cls):
        """
        清除已经完成的进度条
        """
        pass
        for task in cls._echo_plus.tasks:
            if task.finished:
                cls._echo_plus.remove_task(task.id)
            pass

    @classmethod
    @property
    def is_alive(cls):
        return cls._echo_plus.live.is_started

    @classmethod
    def get_unfinished_tasks(cls):
        """
        获取没有运行结束的任务
        """
        list_unfinished = []
        for task in cls._echo_plus.tasks:
            if not task.finished:
                list_unfinished.append(task)
        return list_unfinished

    @staticmethod
    def sub_progress(generator: Optional[Iterable[TT]] = None, title="x_progress", total: Optional[int] = None, visible=True, *args, **kwargs) -> Iterable[TT]:
        """
        返回一个会自动隐藏的子进度条
        """
        kwargs["auto_hide"] = True
        return X_Progress(generator, title, total, *args, visible=visible, **kwargs)

    def sub(self, generator: Optional[Iterable[TT]] = None, title="x_progress", total: Optional[int] = None, visible=True, *args, **kwargs) -> Iterable[TT]:
        """
        using with `with`
        返回一个会自动隐藏的子进度条
        """
        kwargs["auto_hide"] = True
        sub_progress = X_Progress(generator, title, total, *args, visible=visible, **kwargs)
        sub_progress.parent = self
        self.dict_child[sub_progress.task_id] = sub_progress
        return sub_progress

    def __len__(self):
        return (
            self.total
            if self.generator is None
            else (
                self.generator.shape[0]
                if hasattr(self.generator, "shape")
                else len(self.generator) if hasattr(self.generator, "__len__") else self.generator.__length_hint__() if hasattr(self.iterable, "__length_hint__") else getattr(self, "total", None)
            )
        )

    def __contains__(self, item):
        contains = getattr(self.generator, "__contains__", None)
        return contains(item) if contains is not None else item in self.__iter__()

    def __iter__(self):
        """
        只有进入迭代的时候才会启动进度条
        """
        if self.task is not None and self.task_id is not None:
            if not self.task.finished:
                # ! 如果上一个迭代器还没有完成，本次再次启动了一个迭代器，
                # ! 同一时间该迭代对象封装器只能管理一次迭代，多次重复进入迭代会强制关闭上一个迭代
                UserWarning("X_Progress can only manage one iteration at a time, and multiple iterations will be forced to close the previous iteration")
                self._echo_plus.remove_task(self.task_id)

        #! 只有这里能启动 进度条
        self.start()

        self.task_id = self._echo_plus.add_task(self.title, total=self.total)
        self.task = self._echo_plus.tasks[self._echo_plus.task_ids.index(self.task_id)]
        self.task.visible = self.visible

        for i in self.generator:
            yield i
            self.advance()
        # * 结束迭代的处理
        if self.auto_hide:
            # self.task.visible = False
            pass
        else:
            print(str(self))

        #! 每次有一个进度条运行完毕的时候，就需要校验 _progress 是否全部完成，如果完成，就自动停止，避免线程干扰
        X_Progress._echo_plus.remove_task(self.task_id)
        if X_Progress._echo_plus.finished:
            X_Progress._echo_plus.stop()

    def __str__(self) -> str:
        if self.task_id is None:
            return f"Progress [{self.title}] not start"

        task = self.task
        _str = [column.format(task=task) if isinstance(column, str) else column(task) for column in self._echo_plus.columns]
        str_print = " ".join([i.plain if isinstance(i, Text) else "" for i in _str])
        return str_print

    def __enter__(self):
        # * 把process放到外面才能运行同时有多个进度条的存在
        # todo 如果是with进入的，则所有子进度条都应当运行完毕后隐藏起来
        #! 注意 with 启动的回显器不应当拥有一个可迭代对象的输入
        assert self.task is None
        # * 这时候初始化一个task
        self.start()
        self.task_id = self._echo_plus.add_task(self.title, total=self.total)
        self.task = self._echo_plus.tasks[self._echo_plus.task_ids.index(self.task_id)]
        return self

    def __exit__(self, *exc_info):
        # * with语句下，退出时直接判定当前task为finish
        assert self.task is not None
        self.task.finished_time = self.task.elapsed
        self.stop_sub()
        if self.task_id:
            X_Progress._echo_plus.remove_task(self.task_id)
        if X_Progress._echo_plus.finished:
            X_Progress._echo_plus.stop()

        pass

    def stop_sub(self):
        for task_id, progress in self.dict_child.items():
            assert progress.task is not None
            progress.task.finished_time = progress.task.elapsed
            progress.stop_sub()
            del progress
        if X_Progress._echo_plus.finished:
            X_Progress._echo_plus.stop()

    def advance(self, step=1):
        X_Progress._echo_plus.update(self.task_id, advance=step)


class Safe_Progress(Progress):
    """

    Example:
    -------

    with Safe_Progress() as progress:
        task_dataset = progress.add_task(description=f"Total Eval ({len(dataset.list_camera)} camera)", total=dataset.num_camera)
        for camera in dataset.list_camera:

            task_camera = progress.add_task(description=f"eval camera: {camera.name}", total=dataset.num_camera)
            dataloader = Dataloader_Camera(camera)
            for data_i in dataloader:
                ...
                progress.update(task_camera, advance=1)
            progress.remove_task(task_camera)
            progress.update(task_dataset, advance=1)
            ...


    """

    def __init__(self, *args, auto_refresh=False, **kwargs):
        kwargs.setdefault("auto_refresh", auto_refresh)
        super().__init__(
            *args,
            TextColumn("[progress.description]{task.description}"),
            SpinnerColumn(),
            BarColumn(),
            Pretty_Text_PercentageColumn(),
            Pretty_SpeedColumn_item(),
            TimeElapsedColumn(),
            TextColumn("ETA:"),
            TimeRemainingColumn(),
            Pretty_TotalAmountColumn(),
            **kwargs,
        )
