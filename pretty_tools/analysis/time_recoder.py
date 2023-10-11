"""
累加计时器
"""
import time
from typing import Callable, List, Tuple, Optional, Union


class Time_Recoder:
    """
    通用计时器，方便性能占比计时

    .. note::
        注意，这个并不是多线程安全的，请用在单个线程或主线程上

    """

    def __init__(self) -> None:
        pass
        self.dict_item = {}
        self.total_cost = 0

    def register_item(self, item_name: Union[str, List[str]]):
        # todo 这部分以后或许可以用 cython 写，方便减少性能损耗
        if isinstance(item_name, str):
            assert item_name not in self.dict_item
            self.dict_item[item_name] = [0, 0, 0, 0]  # * 分别是开始时间，结束时间，时间求和，调用次数
        elif isinstance(item_name, list):
            for name in item_name:
                assert name not in self.dict_item
                self.dict_item[name] = [0, 0, 0, 0]
        else:
            raise TypeError(f"错误的 item_name 输出类型: {type(item_name)}")

    def register_fn(self, fn: Callable):
        """
        注册一个被计时的函数


        .. code-block:: python
            timer = Time_Recoder()

            @timer.register_fn
            def func_to_be_record():
                ...

        """
        item_name = fn.__name__
        self.register_item(item_name)

        def wrap(*args, **kwargs):
            t_start = time.time()
            self.dict_item[item_name][0] = t_start
            result = fn(*args, **kwargs)
            t_end = time.time()
            self.dict_item[item_name][1] = t_end
            self.dict_item[item_name][2] += t_end - t_start
            return result

        return wrap

    def item_start(self, item_name: str):
        l = self.dict_item[item_name]
        l[0] = time.time()
        pass

    def item_end(self, item_name: str):
        l = self.dict_item[item_name]
        l[1] = time.time()
        l[2] += l[1] - l[0]
        l[3] += 1

    def get_item_cost(self, item_name: str):
        """
        获取计时求和
        """
        return self.dict_item[item_name][2]

    def get_item_ncall(self, item_name: str):
        """
        获取调用次数
        """
        return self.dict_item[item_name][3]

    def total_start(self):
        self.__start_time = time.time()

    def total_end(self):
        self.total_cost = time.time() - self.__start_time
