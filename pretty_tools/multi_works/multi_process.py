import os
import queue
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Lock, Pool, Process, cpu_count
from types import TracebackType
from typing import Any


def handle_error(exception):
    print(f"Error occurred: {exception}")


class BoundProcessPoolExecutor(ProcessPoolExecutor):
    pass

    # todo 之后添加 monitor 功能
    def __init__(
        self,
        max_workers: "int | None" = None,
        qsize: "int | None" = None,
        mp_context=None,
        initializer=None,
        initargs: tuple[Any, ...] = (),
    ) -> None:
        """
        ! 底层解析，默认这个类只能同时运行 max_workers 个进程，但是附加到进程池的任务是非阻塞的，也就是说可以附加无穷个，导致内存被击穿

        现在对其进行修改，使得当进程池附加的任务数量是 workers 的两倍时，自动阻塞

        由于 submit 函数的设计，这种强制对 self._result_queue 进行 get 的操作会使得进程池无法将函数计算值返回给主函数，因此最好不要塞进来一个需要返回值的函数（除非手动放一个 queue)
        """
        if max_workers is None:
            max_workers = cpu_count()
        super().__init__(max_workers, mp_context, initializer, initargs)
        self.max_workers = max_workers
        if qsize is None and max_workers is not None:
            qsize = 2 * max_workers  # 默认用 max_works的两倍
        # self._work_ids = queue.Queue(maxsize=qsize)  # type: ignore  # * 创建任务后，会放到这个队列中，这里通过这个队列实现 进程池满状态时的阻塞 (补充，改这个没用，会在进程池满的时候在这一直等待，应该需要另一个进程来解放该进程)
        self.qsize: int = qsize  # type: ignore

        self.__force_exit = False
        self.__force_exit_ok = Lock()

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type: "type[BaseException] | None", exc_val: "BaseException | None", exc_tb: "TracebackType | None") -> "bool | None":
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __repr__(self) -> str:
        num_pending_works = len(self._pending_work_items)  # * 这个是 当前附加的任务字典

        return f"{self.__class__.__name__}(max_workers={self.max_workers}, qsize={self.qsize}, num_pending_works={num_pending_works})"

    def submit(self, fn, /, *args, **kwargs):
        """
        重写的一个带有阻塞功能的 submit 方法，当队列中已经有任务在执行且全部都未执行完毕，submit 新的任务的时候会阻塞，直到队列中任何一个任务完成

        """
        self.__force_exit_ok.acquire(block=False)
        while True:
            if len(self._pending_work_items) >= self.qsize and not self.__force_exit:
                time.sleep(0.00001)
                # * 未提交之前，不允许 shutdown
            else:
                break

        if self.__force_exit:
            self.__force_exit_ok.release()
            return

        super().submit(fn, *args, **kwargs)
        self.__force_exit_ok.release()

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False):
        if cancel_futures:
            self.__force_exit = True
        self.__force_exit_ok.acquire(block=True)
        self.__force_exit_ok.release()
        super().shutdown(wait=wait, cancel_futures=cancel_futures)
