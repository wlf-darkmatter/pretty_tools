import time
import rich


class X_Timer:
    """
    .. code-block:: python

        >>> from pretty_tools.echo import X_Timer

        >>> timer = X_Timer()
        >>> the_order = ["first", "second", "1-3", "1-4", "last_from_init"]
        >>> last_order = [None, None, "first", "first", "init_time"]

        >>> for _ in range(100):
        >>>     for label, last_label in zip(the_order, last_order):
        >>>         time.sleep(0.01)
        >>>         timer.record(label, last_label=last_label, verbose=True)
        >>>     time.sleep(0.3)


    """

    init_label = "init_time"

    def __init__(self, detail=False):
        """
        Args:
            save_detail (bool): 是否保存详细的时间记录
        """
        self.max_label_len = len(self.init_label)
        self.detail = detail
        self.start()

    def start(self) -> float:
        self.init_time = time.time()
        self.list_item = [self.init_label]
        self.dict_item = {self.init_label: self.init_time}
        self.dict_detail = {}
        return self.init_time

    def record(self, label: str, last_label: str = None, verbose=False) -> float:
        """
        Args:
            label (str): 当前的label
            last_label (str): 用以统计时间差的上一个label，如果为 None 则表示上一个label为自己本身
            verbose (bool): 是否打印时间，注意，频繁打印也非常耗时，4000次大概耗时 23.6秒，约一次耗时 5.9 ms

        Return:
            float: 时间差

        """

        curr_time = time.time()

        if label not in self.list_item:
            self.list_item.append(label)
            if len(label) >= self.max_label_len:
                self.max_label_len = len(label)

        if last_label is None:
            curr_index = self.list_item.index(label)
            last_index = curr_index - 1
            if last_index == 0:
                last_label = self.list_item[-1]
            else:
                last_label = self.list_item[last_index]
            # 默认用记录值的最后一个label作为 last_label
            # 如果最后一个label是自己，说明目前只有自己一个label，
            if last_label == label:
                if label in self.dict_item:
                    # 用自己的自己做差值
                    last_time = self.dict_item[last_label]
                else:
                    # 如果不存在之前的自己，则跟init_time做差值
                    last_time = self.init_time
            else:  # 有其他的标签，跟上一个标签做差值
                last_time = self.dict_item[last_label]
        else:
            last_time = self.dict_item[last_label]
        dt = curr_time - last_time

        if verbose:
            rich.print(f"['{label.ljust(self.max_label_len,' ')}']: {dt:03.4f} s. timed from '{last_label.center(self.max_label_len,' ')}' to '{label.center(self.max_label_len,' ')}'")
            # * 打印 会影响后续的统计时间，这里给消除掉，但是只能影响线性的计算，无法影响指定label的耗时统计

        if self.detail:
            self.dict_detail.setdefault(label, []).append(dt)
            pass
        self.dict_item[label] = time.time()

        return dt

    def save_detail(self, file_path):
        from scipy import stats
        import numpy as np

        # 计算描述统计
        with open(file_path, "w") as f:
            for label, time_list in self.dict_detail.items():
                if len(time_list) > 1:
                    desc = stats.describe(time_list)
                    # 提取平均值和方差
                    u_t, s_t = desc.mean, np.sqrt(desc.variance)
                else:
                    u_t, s_t = time_list[0], 0

                f.write(f"{label.ljust(self.max_label_len,' ')} (count to {str(len(time_list)).ljust(6,' ')}):  {self.__format_time_str(u_t)} ± {self.__format_time_str(s_t)} \n")

    def __format_time_str(self, dt: float):
        if dt > 60:
            return f"{int(dt):d} s"
        elif dt > 10:
            return f"{dt:.1f} s"
        elif dt > 1:
            return f"{dt:.2f} s"
        elif dt > 1e-1:
            return f"{int(dt*1000):d} ms"
        elif dt > 1e-2:
            return f"{dt*1000:.1f} ms"
        elif dt > 1e-3:
            return f"{dt*1000:.2f} ms"
        elif dt > 1e-4:
            return f"{int(dt*1000000):d} μs"
        elif dt > 1e-5:
            return f"{dt*1000000:.1f} μs"
        elif dt > 1e-6:
            return f"{dt*1000000:.2f} μs"
