import time

from pretty_tools.echo.x_progress import X_Progress


class Test_R_Progress:
    """
    这个的多线程还没测试过
    """

    # * 定义一个无穷长度的迭代器（无法统计长度）, 但是不超过1000
    @staticmethod
    def inf_iterator():
        i = 0
        for i in range(300):
            yield i
            i += 1

    @staticmethod
    def test_R_Progress_normal():
        """
        总长度已知进度条
        """
        for i in X_Progress(range(300), "test_R_Progress_normal"):
            time.sleep(0.01)
        assert not X_Progress.is_alive

    @staticmethod
    def test_R_Progress_inf():
        """
        总长度未知进度条
        """
        for i in X_Progress(Test_R_Progress.inf_iterator(), "test_R_Progress_inf"):
            time.sleep(0.01)
        assert not X_Progress.is_alive

    @staticmethod
    def test_multi_level():
        """
        多层级进度条
        """
        for i in X_Progress(range(4), "Main Progress"):
            for j in X_Progress(range(4), f"Minor Progress {i}"):
                for k in X_Progress(range(50), f"Mini Progress {i}-{j}"):
                    time.sleep(0.005)
        assert not X_Progress.is_alive

    @staticmethod
    def test_multi_level_plus():
        """
        多层级进度条, 子进度条会自动隐藏

        """
        for i in X_Progress(range(3), "Main Progress"):
            for j in X_Progress.sub_progress(range(4), f"Minor Progress {i}"):
                for k in X_Progress.sub_progress(range(50), f"Mini Progress {i}-{j}"):
                    time.sleep(0.005)
        assert not X_Progress.is_alive

    @staticmethod
    def dont_test_break():
        """
        ! 这个示例说明 for 开启的一个循环体 无法判断进度条是否被break，X_Progress会继续保持运行，所以该测试必定失败

        """
        for i in X_Progress(range(100), "check break"):
            pass
            if i == 50:
                break
            time.sleep(0.01)
        assert not X_Progress.is_alive

    @staticmethod
    def test_with():
        with X_Progress(title="Big Progress") as progress:
            flag_break = False
            for i in range(100):
                progress.advance()
                for j in progress.sub(range(50), "临时子进度条"):
                    time.sleep(0.001)
                    if i == 50:
                        flag_break = True
                        break
                if flag_break:
                    break
            print("break out")
            pass
        pass
        assert not X_Progress.is_alive

    @staticmethod
    def test_different_progress():
        """
        测试同时存在两个不同的进度条
        """
        from rich.progress import track

        print(f"进度条是否还在运行: {X_Progress.is_alive}")

        # * 测试一，先创建了一个活动的 track 进度条，自己后创建的进度条会依附于前者运行
        A = X_Progress(range(100), "progress A")  # 这里只是初始化，并没有start一个live
        for i in track(range(10)):
            # print(f"i={i}")
            print("------------------")
            for j in A:
                time.sleep(0.001)
        pass
        # * 测试二，先创建了一个活动的 X_Progress 进度条，后创建的进度条会依附于前者运行
        # ? 测试二是不能运行的，因为存在了一个自己设计的进度条，一直在运行，内置的track会试图启动一个新的live，但是和自己设计的冲突了，所以无法运行
        assert not X_Progress.is_alive

        # for i in A:
        #     for j in track(range(10)):
        #         time.sleep(0.001)


class Test_Runtime_info:
    """
    运行时打印资源占用信息
    """

    @staticmethod
    def test_print():
        from pretty_tools.echo.echo_resource import Echo_Resource

        with Echo_Resource(live=True) as echo_cpu:
            for i in X_Progress(range(1000)):
                # echo_cpu.print_cpu()  #* 频繁打印存在一定的性能损耗
                time.sleep(0.001)
            pass

    @staticmethod
    def test_timer():
        from pretty_tools.echo import X_Timer

        timer = X_Timer()

        the_order = ["first", "second", "1-3", "1-4", "last_from_init"]
        last_order = [None, None, "first", "first", "init_time"]

        for _ in range(100):
            for label, last_label in zip(the_order, last_order):
                time.sleep(0.0001)
                timer.record(label, last_label=last_label, verbose=True)
            time.sleep(0.0003)

    pass


if __name__ == "__main__":
    # test_R_Progress_normal()
    # test_R_Progress_inf()  #! nimble电脑中，不打印输出的话，速度维持在 98 it/s左右，打印输出的话，维持在 82 it/s
    # Test_R_Progress.test_multi_level()
    # Test_R_Progress.test_break()
    # X_Progress.clean()

    # Test_R_Progress.test_multi_level_plus()
    # Test_R_Progress.test_with()
    # print(f"进度条是否还在运行: {X_Progress.is_alive}")
    # Test_R_Progress.test_different_progress()
    print(f"进度条是否还在运行: {X_Progress.is_alive}")

    Test_Runtime_info.test_print()
