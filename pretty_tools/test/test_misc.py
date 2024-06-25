import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from pretty_tools.datastruct.misc import (np_get_gt_match_from_id_0,
                                          py_get_gt_match_from_id)


class Test_get_match_gt:
    """ """

    def setup_method(self):
        np.random.seed(1213)
        # * 自动生成100个可以匹配的id对
        self.list_id_a: List[np.ndarray] = []
        self.list_id_b: List[np.ndarray] = []
        self.list_both_id: List[np.ndarray] = []
        self.list_gt_match: List[np.ndarray] = []
        test_max = 50
        self.num_test_calc = 1000
        for _ in range(10000):
            # 被匹配的数量大致上是 0~50 个
            n_a = np.random.randint(1, 50)
            n_b = np.random.randint(1, 50)
            if np.min([n_a, n_b]) != 0:
                n_both_id = np.random.randint(0, np.min([n_a, n_b]))
            else:
                n_both_id = 0
            # 随机生成 n_both_id 个 共同id号
            both_id = np.random.choice(np.arange(2 * test_max), n_both_id, replace=False)  # * 最差情况有 test_max + test_max 个完全不同的数组
            self.list_both_id.append(both_id)
            tmp_a = np.array(both_id.tolist() + np.arange(2 * test_max, 2 * test_max + n_a - n_both_id).tolist())
            tmp_b = np.array(both_id.tolist() + np.arange(3 * test_max, 3 * test_max + n_b - n_both_id).tolist())
            # * 随机生成打乱的索引
            shuffle_a = np.random.choice(range(n_a), n_a, replace=False)  # * 前n_both_id个位置作为真值
            shuffle_b = np.random.choice(range(n_b), n_b, replace=False)  # * 前n_both_id个位置作为真值
            new_tmp_a = np.zeros_like(tmp_a)
            new_tmp_b = np.zeros_like(tmp_b)
            new_tmp_a[shuffle_a] = tmp_a
            new_tmp_b[shuffle_b] = tmp_b

            self.list_id_a.append(new_tmp_a)
            self.list_id_b.append(new_tmp_b)
            # * 真值就是shuffle_a和shuffle_b的前n_both_id个元素
            match_gt = np.stack([shuffle_a[:n_both_id], shuffle_b[:n_both_id]], axis=1)
            self.list_gt_match.append(match_gt)
        pass

    def test_auto_generate(self):
        tmp_a = self.list_id_a[0]
        tmp_b = self.list_id_b[0]
        match_gt = self.list_gt_match[0]
        both_id = self.list_both_id[0]
        assert (tmp_a[match_gt[:, 0]] == tmp_b[match_gt[:, 1]]).all()

        assert sorted(np.intersect1d(tmp_a, tmp_b)) == sorted(both_id)

    def test_one_calc(self):
        from pretty_tools.datastruct.misc import get_gt_match_from_id

        tmp_a = self.list_id_a[0]
        tmp_b = self.list_id_b[0]
        tmp_gt = self.list_gt_match[0]

        np_match = get_gt_match_from_id(tmp_a, tmp_b)

        # * 哈希集合校验
        set_tmp_gt = set([tuple(i) for i in tmp_gt])
        set_tmp_calc = set([tuple(i) for i in np_match])
        assert set_tmp_gt == set_tmp_calc

        pass

    def test_all_calc_np(self):
        from pretty_tools.datastruct.misc import np_get_gt_match_from_id_0

        # * 验证数量为 self.num_test_calc 个
        i = 0
        for a, b, gt in zip(self.list_id_a, self.list_id_b, self.list_gt_match):
            list_match = np_get_gt_match_from_id_0(a, b)
            set_tmp_gt = set([tuple(i) for i in gt])
            set_tmp_calc = set([tuple(i) for i in list_match])
            assert set_tmp_gt == set_tmp_calc
            # --------------------------------------------
            i += 1
            if i == self.num_test_calc:
                break

    def test_all_calc_py(self):
        from pretty_tools.datastruct.misc import py_get_gt_match_from_id

        # * 验证数量为 self.num_test_calc 个
        i = 0
        for a, b, gt in zip(self.list_id_a, self.list_id_b, self.list_gt_match):
            list_match, _ = py_get_gt_match_from_id(a.tolist(), b.tolist())
            set_tmp_gt = set([tuple(i) for i in gt])
            set_tmp_calc = set([tuple(i) for i in list_match])
            assert set_tmp_gt == set_tmp_calc
            # --------------------------------------------
            i += 1
            if i == self.num_test_calc:
                break

    def test_all_calc_cy(self):
        from pretty_tools.datastruct.misc import cy_get_gt_match_from_id

        i = 0
        for a, b, gt in zip(self.list_id_a, self.list_id_b, self.list_gt_match):
            list_match, _ = cy_get_gt_match_from_id(a, b)
            set_tmp_gt = set([tuple(i) for i in gt])
            set_tmp_calc = set([tuple(i) for i in list_match])
            assert set_tmp_gt == set_tmp_calc
            # --------------------------------------------
            i += 1
            if i == self.num_test_calc:
                break

    def test_speed(self):
        from pretty_tools.datastruct.misc import cy_get_gt_match_from_id

        time_start = time.time()
        for a, b in zip(self.list_id_a, self.list_id_b):
            np_get_gt_match_from_id_0(a, b)
            pass
        time_end = time.time()
        time_cost_np = time_end - time_start
        # --------------------------------------------
        # 测试 python list的实现速度，提前将numpy转换为list
        py_list_id_a = [i.tolist() for i in self.list_id_a]
        py_list_id_b = [i.tolist() for i in self.list_id_b]
        time_start = time.time()
        for a, b in zip(py_list_id_a, py_list_id_b):
            py_get_gt_match_from_id(a, b)
            pass
        time_end = time.time()
        time_cost_py = time_end - time_start
        # --------------------------------------------
        time_start = time.time()
        for a, b in zip(self.list_id_a, self.list_id_b):
            list_match, _ = cy_get_gt_match_from_id(a, b)
            pass
        time_end = time.time()
        time_cost_cy = time_end - time_start
        # --------------------------------------------
        path_log = Path(__file__).with_name("log")
        path_log.mkdir(exist_ok=True)
        with open(path_log.joinpath(self.__class__.__name__ + "_time_cost.txt"), "w") as f:
            f.write(f"np_get_gt_match_from_id() : {time_cost_np:.4f} s\n")
            f.write(f"py_get_gt_match_from_id() : {time_cost_py:.4f} s\n")
            f.write(f"cy_get_gt_match_from_id() : {time_cost_cy:.4f} s\n")
            # f.write(f'nb_get_gt_match_from_id() : {time_cost_nb:.4f} s\n')
        #! Cython 编译出来的确实更快，是python的三倍
        """
        np_get_gt_match_from_id() : 0.7492 s
        py_get_gt_match_from_id() : 0.0691 s
        cy_get_gt_match_from_id() : 0.0227 s
        nb_get_gt_match_from_id() : 0.2828 s
        """


if __name__ == "__main__":
    pytest.main(
        [
            "-s",
            "-l",
            # "test_misc.py",
            __file__
        ]
    )
