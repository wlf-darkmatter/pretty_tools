"""
.. note::
    **Stable** 模块，长期支持
    毕竟都写成了 Cython 了，不可能不用的


由于 Cython 无法识别函数的参数，因此需要在文档中自己添加参数的类型和顺序
"""
# cython:language_level=3

# --------------------------------------------------------
# ContionTrack
# Copyright (c) 2023 BIT
# Licensed under The MIT License [see LICENSE for details]
# Written by Wang Lingfeng
# --------------------------------------------------------

cimport cython
import numpy as np
cimport numpy as cnp


@cython.boundscheck(False)
@cython.wraparound(False)
def index_value_2d_(
    cnp.ndarray input,
):
    cdef unsigned int N1 = input.shape[0]
    cdef unsigned int N2 = input.shape[1]
    cdef cnp.ndarray[cnp.float_t, ndim=2] index_value = np.zeros((N1 * N2, 3), dtype=float)

    cdef unsigned int i1
    cdef unsigned int i2
    cdef int count = 0

    for i1 in range(N1):
        for i2 in range(N2):
            index_value[count][0] = i1
            index_value[count][1] = i2
            index_value[count][2] = input[i1, i2]
            count = count + 1
    return index_value

@cython.boundscheck(False)
@cython.wraparound(False)
def index_value_2d_nonzero(
    cnp.ndarray input,
):
    cdef unsigned int N1 = input.shape[0]
    cdef unsigned int N2 = input.shape[1]
    cdef cnp.ndarray[cnp.float_t, ndim=2] index_value = np.zeros((N1 * N2, 3), dtype=float)

    cdef unsigned int i1
    cdef unsigned int i2
    cdef int count = 0

    for i1 in range(N1):
        for i2 in range(N2):
            if input[i1, i2] !=0:
                index_value[count][0] = i1
                index_value[count][1] = i2
                index_value[count][2] = input[i1, i2]
                count = count + 1
    return index_value[:count]

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_bisect_right(
    cnp.ndarray a,
    float x,
    int lo,
    int hi,
):
    """
    Optional args :code:`lo` (default 0) and :code:`hi` (default :code:`len(a)` ) bound the slice of :code:`a` to be searched.

    return:
        index where to insert item :code:`x` in list :code:`a`, assuming :code:`a` is **sorted**.
    """
    # lo 默认为0
    cdef unsigned int mid
    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo

def cy_bisect_right_array(
    cnp.ndarray a,
    cnp.ndarray x,
    int lo,
    int hi,
    ):
    cdef unsigned int mid
    cdef cnp.ndarray[cnp.int64_t, ndim=1] lo_array = np.zeros(len(x), dtype=int)
    cdef int i = 0
    cdef int _lo = lo
    cdef int _hi = hi


    for i in range(len(x)):
        x_i = x[i]
        while _lo < _hi:
            mid = (_lo + _hi) // 2
            if x_i < a[mid]:
                _hi = mid
            else:
                _lo = mid + 1
        lo_array[i] = _lo
        _lo = lo
        _hi = hi
    return lo_array


@cython.boundscheck(False)
@cython.wraparound(False)
def cy_bisect_left(
    cnp.ndarray a,
    float x,
    int lo,
    int hi,
):
    """
    Optional args :code:`lo` (default 0) and :code:`hi` (default :code:`len(a)` ) bound the slice of :code:`a` to be searched.

    return:
        index where to insert item :code:`x` in list :code:`a`, assuming :code:`a` is **sorted**.
    """
    # lo 默认为0
    cdef unsigned int mid
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo

def cy_bisect_left_array(
    cnp.ndarray a,
    cnp.ndarray x,
    int lo,
    int hi,
    ):
    cdef unsigned int mid
    cdef cnp.ndarray[cnp.int64_t, ndim=1] lo_array = np.zeros(len(x), dtype=int)
    cdef int i = 0
    cdef int _lo = lo
    cdef int _hi = hi

    for i in range(len(x)):
        x_i = x[i]
        while _lo < _hi:
            mid = (_lo + _hi) // 2
            if a[mid] < x_i:
                _lo = mid + 1
            else:
                _hi = mid
        lo_array[i] = _lo
        _lo = lo
        _hi = hi
    return lo_array
#---------------------------- One-Hot -----------------------------------

def cy_onehot_by_index(
    cnp.ndarray array_index,
    int depth_feature,
):
    """
    生成 one-hot 编码

    args:
        array_index: 一维数组, 传入 **One-Hot** 编码的位置, 该位置不应当大于 :code:`depth_feature`
        depth_feature: ont-hot 的编码深度

    """
    cdef unsigned int N = len(array_index)
    cdef cnp.ndarray[cnp.float_t, ndim=2] onehot = np.zeros((N, depth_feature), dtype=float)

    cdef int i = 0
    for i in range(N):
        onehot[i][array_index[i]] = 1
    return onehot

#! 用range的时候不能优化边缘检测
# @cython.boundscheck(False)
# @cython.wraparound(False)
def cy_onehot_by_cumulate(
    cnp.ndarray array_cumsum,
    int depth_feature,
):
    """
    生成 one-hot 编码, 采用连续性编码，编码深度由 depth_feature 指定，但是当 :code:`depth_feature < len(array_cumsum) - 1` 时，采用 :code:`len(array_cumsum) - 1` 作为编码深度 `array_cumsum` 指定的深度

    args:
        array_cumsum: 一维数组, 传入 **One-Hot** 连续性编码的索引位置, 连续编码的顺序从1递增， :code:`array_cumsum[0] == 0`

    """
    cdef unsigned int N = array_cumsum[-1]
    cdef unsigned int _depth_feature = len(array_cumsum) - 1
    if depth_feature < _depth_feature:
        depth_feature = _depth_feature
    cdef cnp.ndarray[cnp.float_t, ndim=2] onehot = np.zeros((N, depth_feature), dtype=float)

    cdef int i = 0
    cdef int j_index_compare = 0
    for i in range(N):
        if i >= array_cumsum[j_index_compare + 1]:
            j_index_compare += 1
        onehot[i][j_index_compare] = 1
    return onehot

#---------------------------- cumsum 的排序自动生成 -----------------------------------

@cython.boundscheck(False)
def cy_cum_to_index(
    cnp.ndarray array_cumsum,
):
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
    cdef unsigned int M = array_cumsum[-1]
    cdef unsigned int i = 0
    cdef unsigned int j = 0
    cdef unsigned int offset = 0

    cdef cnp.ndarray[cnp.float_t, ndim=1] index_value = np.zeros((M, ), dtype=float)

    if array_cumsum[0] == 0:
        offset = -1
    for i in range(M):
        if i == array_cumsum[j]:
            j += 1
        index_value[i]= j + offset

    return index_value

