from typing import List

import numpy as np

from ._C_misc import cy_get_gt_match_from_id


def py_get_gt_match_from_id(id_a: List, id_b: List) -> tuple[list, list]:
    list_match = []
    list_id = []
    for i, id0 in enumerate(id_a):
        try:
            list_match.append((i, id_b.index(id0)))
            list_id.append(id0)
        except ValueError:
            pass

    return list_match, list_id


def np_get_gt_match_from_id_0(np_id_a: np.ndarray, np_id_b: np.ndarray) -> List:
    """
    从两个id数组中，得到匹配的gt
    np_id_0: 第一个id数组
    np_id_1: 第二个id数组

    #! 不用测试numpy了，怎么都没有python原来的快，主要是查找一个元素的索引的速度太慢了
    """
    list_match = []

    # * 先找到相同的元素
    both_id = np.intersect1d(np_id_a, np_id_b)
    # * 然后找对应的下标
    for i in both_id:
        list_match.append([*np.argwhere(np_id_a == i)[0], *np.argwhere(np_id_b == i)[0]])

    return list_match


def get_gt_match_from_id(id_a, id_b, method: int = 1):
    """
    np_id_a <List>
    np_id_b <List>

    np_id_a <np.ndarray>
    np_id_b <np.ndarray>

    不用测试numpy了，怎么都没有python原来的快，主要是查找一个元素的索引的速度太慢了
    """
    if method == 1:
        return np_get_gt_match_from_id_0(id_a, id_b)
    elif method == 3:
        assert type(id_a) is List and type(id_b) is List
        return py_get_gt_match_from_id(id_a, id_b)
    else:
        raise NotImplementedError()
