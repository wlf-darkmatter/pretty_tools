import queue
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from types import TracebackType
from typing import Any, Callable

from ..echo import Echo_Resource


class BoundThreadPoolExecutor(ThreadPoolExecutor):
    """
    对ThreadPoolExecutor 进行重写，给队列设置边界，否则会存在大量的内存占用
    """

    def __init__(
        self,
        max_workers: "int | None" = None,
        qsize: "int | None" = None,
        monitor=False,
        thread_name_prefix: str = "",
        initializer: "Callable[..., object] | None" = None,
        initargs: "tuple[Any, ...]" = ...,
    ) -> None:
        """
        monitor: 是否开启监控
        """
        if max_workers is None:
            max_workers = cpu_count()
        super().__init__(max_workers, thread_name_prefix, initializer, initargs)
        self.max_workers = max_workers

        if monitor:
            self.monitor = Echo_Resource()
        else:
            self.monitor = None

        if qsize is None and max_workers is not None:
            qsize = 2 * max_workers  # 默认用 max_works的两倍
        self.qsize = qsize
        self._work_queue = queue.Queue(qsize)  # type: ignore

    def __enter__(self):
        if self.monitor:
            self.monitor.__enter__()
        return super().__enter__()

    def __exit__(self, exc_type: "type[BaseException] | None", exc_val: "BaseException | None", exc_tb: "TracebackType | None") -> "bool | None":
        if self.monitor:
            self.monitor.__exit__(exc_type, exc_val, exc_tb)
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_workers={self.max_workers}, qsize={self.qsize})"
