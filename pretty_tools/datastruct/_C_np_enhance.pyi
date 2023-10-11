import numpy as np

def index_value_2d_(input: np.ndarray) -> np.ndarray:
    """
    input (N, M) 大小的矩阵
    返回 (N * M, 3) 大小的矩阵， 第0列为行索引，第1列为列索引，第2列为值
    """
    ...

def index_value_2d_nonzero(input: np.ndarray) -> np.ndarray: ...
def cy_bisect_right_array(a: np.ndarray, x: np.ndarray, lo: int, hi: int) -> np.ndarray: ...
def cy_bisect_right(a: np.ndarray, x: float, lo: int, hi: int) -> int:
    """
    Default args lo (default 0) and hi (default len(a)) bound the slice of a to be searched

    Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(i, x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """
    ...

def cy_bisect_left_array(a: np.ndarray, x: np.ndarray, lo: int, hi: int) -> np.ndarray: ...
def cy_bisect_left(a: np.ndarray, x: float, lo: int, hi: int) -> int:
    """
    Default args lo (default 0) and hi (default len(a)) bound the slice of a to be searched

    Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(i, x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """
    ...

# ---------------------------- One-Hot -----------------------------------
def cy_onehot_by_index(array_index: np.ndarray, depth_feature: int) -> np.ndarray:
    """
    生成 one-hot 编码

    args:
        array_index: 一维数组, 传入 **One-Hot** 编码的位置, 该位置不应当大于 :code:`depth_feature`
        depth_feature: ont-hot 的编码深度

    """
    ...

def cy_onehot_by_cumulate(array_cumsum: np.ndarray, depth_feature: int) -> np.ndarray:
    """
    生成 one-hot 编码, 采用连续性编码，编码深度由 depth_feature 指定，但是当 :code:`depth_feature < len(array_cumsum) - 1` 时，采用 :code:`len(array_cumsum) - 1` 作为编码深度array_cumsum` 指定的深度

    args:
        array_cumsum: 一维数组, 传入 **One-Hot** 连续性编码的索引位置, 连续编码的顺序从1递增， :code:`array_cumsum[0] == 0`

    """
    ...

# ---------------------------- cumsum 的排序自动生成 -----------------------------------

def cy_cum_sort(array_cum: np.ndarray) -> np.ndarray:
    """
    生成 index 索引号, 采用连续性编码

    args:
        array_cumsum: 一维数组, 传入 连续性编码的索引位置, 连续编码的顺序从1递增， :code:`array_cumsum[0] == 0`


    """
    ...

def cy_cum_to_index(array_cumsum: np.ndarray) -> np.ndarray:
    """
    生成 index 索引号, 采用连续性编码

    args:
        array_cumsum: 一维数组, 传入连续性编码的索引位置, 连续编码的顺序从1递增， :code:`array_cumsum[0] == 0`

    .. note::

        :code:`array_cumsum` 必须是已经排序好的，且第一个元素如果不为 0 则会自动插入一个 0，方便统一代码的运行。


    Example
    -------

    .. code::

        import numpy as np

        a = np.array([0, 2, 5, 9])
        cy_cum_to_index(a)

        >>> array([0, 0, 1, 1, 1, 2, 2, 2, 2])

    return:
        np.ndarray(cnp.float_t, ndim=1)

    """
    ...
