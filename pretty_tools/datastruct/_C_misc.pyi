import numpy as np

def cy_get_gt_match_from_id(np_id_a: np.ndarray, np_id_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    从两个id数组中，得到匹配的gt 的索引号
    np_id_a: 第一个id数组
    np_id_b: 第二个id数组

    Return:
    第一个为 (m, 2) 的数组，表示匹配元素在 np_id_a 和 np_id_b 中的索引号
    第二个为 (m, ) 的数组，表示匹配元素的 ID 号
    """
    ...
