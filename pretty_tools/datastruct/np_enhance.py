from __future__ import annotations

from typing import Any, Generic, Iterable, Optional, Sequence, Tuple, TypeAlias, TypeVar, Union

import numpy as np
from numpy._typing import NDArray
from scipy import sparse

from ._C_np_enhance import (
    cy_bisect_left,
    cy_bisect_left_array,
    cy_bisect_right,
    cy_bisect_right_array,
    cy_cum_to_index,
    cy_onehot_by_cumulate,
    cy_onehot_by_index,
    index_value_2d_,
    index_value_2d_nonzero,
)
from .cython_bbox import cy_bbox_overlaps_area, cy_bbox_overlaps_iou

T_arrayType = TypeVar("T_arrayType", np.ndarray, sparse.spmatrix)
T = TypeVar("T")

np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # type: ignore
print(f"Waring: 在 {__file__} 定义了 numpy 打印不换行")
array_Type = Union[np.ndarray, sparse.spmatrix]


def enhance_stack(arrays, axis=0) -> np.ndarray:
    pass
    if len(arrays) == 0:
        return np.array([])
    else:
        return np.stack(arrays, axis=axis)


def convert_listadj_to_edgevalue(list_adj: list):
    """
    Args:
        list_adj (): 一个列表，每个元素是一个邻接矩阵，其中每个元素的行和列都是相同的。必须有正确的尺寸定义

    Return:
        edge_index, values
    """
    # * 如果是列表，必然通过稀疏矩阵的方式进行可视化
    list_np_adj = [convert_to_numpy(adj_i, sparse_shape=adj_i.shape) for adj_i in list_adj]

    inc = np.matrix(np.cumsum(np.stack([(0, 0)] + [adj_i.shape for adj_i in list_np_adj]), axis=0).T)
    list_edge_index = [np.stack(adj_i.nonzero()) + inc[:, i] for i, adj_i in enumerate(list_np_adj)]
    list_value = [adj_i[adj_i.nonzero()] for adj_i in list_np_adj]
    values = np.concatenate(list_value)
    edge_index = np.array(np.concatenate(list_edge_index, axis=1))
    return edge_index, values


def convert_to_numpy(input_array, sparse_shape: Optional[Union[tuple, list]] = None) -> np.ndarray:  # type: ignore
    """
    Convert a Anything to a numpy array.

    Args:
        input_array (Any)
        sparse_shape (Optional[Union[tuple, list]]): 可选的，如果输入的是稀疏矩阵的话，需要这个来确定尺寸

    """
    if input_array is None:
        return None  # type: ignore

    if isinstance(input_array, np.ndarray):
        return input_array
    if str(type(input_array)) == "<class 'torch.Tensor'>":
        try:
            import torch

            if isinstance(input_array, torch.Tensor):
                if input_array.is_sparse:
                    from pretty_tools.datastruct.torch_enhance import Utils_Sparse as torch_utils_sparse

                    return torch_utils_sparse.torch_sparse_to_scipy_sparse(input_array, shape=sparse_shape)
                else:
                    return input_array.detach().cpu().numpy()
            elif isinstance(input_array, np.ndarray):
                return input_array
            else:
                return input_array.numpy()
        except Exception as e:
            print("输入的图像类型为 {type(input_array)}, 并未测试过")
            print(e.__repr__())
            raise TypeError


class block(Generic[T_arrayType]):
    """
    用于将多个矩阵拼接成一个大矩阵
    todo 还有比较多的切片操作没有实现

    支持 稀疏矩阵 csr格式
    merge_from 操作没有针对稀疏矩阵进行优化

    如果传入的是稀疏矩阵，则对原始的data进行操作，会同步到block（本质上block的提取就是对data的切片操作）
    但是对block的操作不会同步到data上，因为对稀疏矩阵的切片操作本质上就产生了一次拷贝

    如果传入的是稠密矩阵，则会对data进行一次切片，应该是一个视图，同样，对block的操作会同步到data上
    """

    @classmethod
    def from_block(cls, data: list) -> block:
        raise NotImplementedError()
        pass

    def __repr__(self) -> str:
        return f"block(data_size={self.data.shape}, block_size={self.shape}, block_slice={self.index_slice})"

    @property
    def block_size(self) -> tuple[int, ...]:
        return tuple(len(i) - 1 for i in self.index_slice)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.block_size

    @classmethod
    def merge_from(cls, data: Iterable[T_arrayType], index_slice: Sequence[Union[np.ndarray, Sequence]], subblock_shape=(0,)) -> block[T_arrayType]:
        # todo 这里全部按照稠密矩阵进行计算，没有针对稀疏进行优化
        #! 只能支持三维以下的操作
        count = 0
        list_subblock = []
        np_shape = np.array(index_slice)
        ndim = np_shape.shape[1]
        assert ndim <= 3, "只能支持三维以下的操作"
        if subblock_shape == (0,):
            subblock_shape = np.max(np_shape, axis=0) + 1  # * 没有规定最大尺度，就取给出的索引中的最大尺度作为 subblock的尺寸

        list_len = [[0 for _ in range(dim)] for dim in subblock_shape]
        for sub_block, index in zip(data, index_slice):
            count += 1
            for dim, dim_l in enumerate(index):
                if list_len[dim][dim_l] == 0:
                    list_len[dim][dim_l] = sub_block.shape[dim]
                else:
                    assert list_len[dim][dim_l] == sub_block.shape[dim]
        assert count == len(index_slice), "输入的矩阵个数必须和数据位置索引列表元素个数保持一致"

        list_np_len = [np.cumsum([0] + dim_len) for dim_len in list_len]
        np_block = np.zeros([i[-1] for i in list_np_len], dtype=float)
        for sub_block, index in zip(data, index_slice):
            if ndim == 1:
                np_block[list_np_len[0][index[0]] : list_np_len[0][index[0] + 1]] = sub_block
            elif ndim == 2:
                np_block[list_np_len[0][index[0]] : list_np_len[0][index[0] + 1], list_np_len[1][index[0]] : list_np_len[1][index[0] + 1]] = sub_block
            elif ndim == 3:
                np_block[
                    list_np_len[0][index[0]] : list_np_len[0][index[0] + 1], list_np_len[1][index[0]] : list_np_len[1][index[0] + 1], list_np_len[2][index[0]] : list_np_len[2][index[0] + 1]
                ] = sub_block
                pass

        return block(np_block, list_np_len)  # type: ignore

    def __init__(self, data: T_arrayType, index_slice: Sequence[Union[np.ndarray, Sequence]]):
        self.data = data
        self.ndim = len(index_slice)  # * 维度数由 index_slice 决定
        assert self.ndim <= 3, f"ndim={self.ndim}>3，暂不支持维度数大于3的分块矩阵"
        assert self.ndim == self.data.ndim, "切分维度要和数据维度统一"  # type: ignore
        self.__subblock = None  # * 初始化为 None, 用于判断是否已经进行了切分，如果是稀疏矩阵，则这个为None

        self.num_piece = []  # * 每个轴被切分的个数
        if isinstance(data, np.ndarray):
            self.type = np.ndarray
        elif isinstance(data, sparse.spmatrix):
            self.type = sparse.spmatrix
        else:
            raise TypeError("不支持的数据类型")
        self.set_index_slice(index_slice)
        #! 如果是稀疏矩阵，则每次都要手动进行一次切片操作

    def set_index_slice(self, index_slice: Sequence[Union[np.ndarray, Sequence]]):
        """
        可以被外界调用，用于设立分块方式
        """
        # * ----------------------------- 性质判定 -----------------------------------
        self.index_slice: list[list[int]] = [[j for j in i] for i in index_slice]
        for a in range(self.ndim):
            if self.index_slice[a][0] != 0:
                self.index_slice[a] = [0] + self.index_slice[a]
            assert self.index_slice[a][0] == 0
            if self.index_slice[a][-1] != self.data.shape[a]:
                self.index_slice[a].append(self.data.shape[a])
            assert self.index_slice[a][-1] == self.data.shape[a]
        assert len(index_slice) == self.ndim, "不允许改变分块维度"
        for l in self.index_slice:
            tmp_l = np.array(l)
            assert ((tmp_l[1:] - tmp_l[:-1]) >= 0).all(), "列表必须是升序"

        self.num_piece = list(len(i) - 1 for i in self.index_slice)  # * 每个轴被切分的个数

        # * ----------------------------- 开始切片 -----------------------------------
        if self.type == np.ndarray:
            self.__subblock = np.zeros(self.num_piece, dtype=np.object_)
            # * 这里会有多层for循环，但是主要取决于到底有几层维度，由于维度本身有限，就不写成迭代形式了，这里的代码将 ndim限制在 3 以内
            for i in range(self.num_piece[0]):
                #! 如果是稀疏矩阵，则这里的切分将会产生一个拷贝
                i_block = self.data[self.index_slice[0][i] : self.index_slice[0][i + 1]]  # type: ignore
                if self.ndim >= 2:
                    for ii in range(self.num_piece[1]):
                        ii_block = i_block[:, self.index_slice[1][ii] : self.index_slice[1][ii + 1]]
                        if self.ndim >= 3:
                            for iii in range(self.num_piece[2]):
                                iii_block = ii_block[:, :, self.index_slice[2][iii] : self.index_slice[2][iii + 1]]
                                self.__subblock[i, ii, iii] = iii_block
                        else:
                            self.__subblock[i, ii] = ii_block
                else:
                    self.__subblock[i] = i_block

    def __getitem__(self, *args) -> T_arrayType:
        #! 如果载入的是稀疏矩阵，则每次提取一个子块都会进行一次切片操作
        if self.__subblock is not None:
            return self.__subblock[*(args[0])]  # type: ignore
        else:
            str_cut = []
            for inds, slices in zip(args[0], self.index_slice):
                cut = f"{slices[inds]}:{slices[inds+1]}"
                str_cut.append(cut)
            str_cut = ",".join(str_cut)
            return eval(f"self.data[{str_cut}]")

    def __setitem__(self, args, target):
        if self.ndim >= 1:
            index_dim0 = (self.index_slice[0][args[0]], self.index_slice[0][args[0] + 1])
            if self.ndim >= 2:
                index_dim1 = (self.index_slice[1][args[1]], self.index_slice[1][args[1] + 1])
                if self.ndim >= 3:
                    index_dim2 = (self.index_slice[2][args[2]], self.index_slice[2][args[2] + 1])
                    self.data[index_dim0[0] : index_dim0[1], index_dim1[0] : index_dim1[1], index_dim2[0] : index_dim2[1]] = target  # type: ignore
                else:
                    self.data[index_dim0[0] : index_dim0[1], index_dim1[0] : index_dim1[1]] = target  # type: ignore
            else:
                self.data[index_dim0[0] : index_dim0[1]] = target  # type: ignore


def index_convert_to_block(index_slice, index):
    """
    索引快速转换, 输入 大矩阵的索引，返回块索引和 块内索引
    """
    index_block = []
    index_inner = []
    assert len(index) == len(index_slice), "索引转换的索引列表数必须和块维度统一"
    # todo 大量输入的时候性能需要优化
    for sub_cumsum, index_dim in zip(index_slice, index):
        assert sub_cumsum[0] == 0, "index_slice 的每一维度 必须是从0开始的"
        sub_cumsum = np.array(sub_cumsum)
        index_block_dim = np.array([bisect_right(sub_cumsum, i) - 1 for i in index_dim])
        index_inner_dim = index_dim - sub_cumsum[index_block_dim]
        index_block.append(index_block_dim)
        index_inner.append(index_inner_dim)
        pass

    return np.array(index_block), np.array(index_inner)


def index_convert_to_combine(index_slice, index_block, index_inner):
    """
    索引快速转换, 输入两个索引列表，第一个是 块索引，第二个是 块中的索引，返回大矩阵的索引
    """
    index_combine = []
    assert len(index_block) == len(index_slice)
    assert len(index_inner) == len(index_slice)
    for sub_cumsum, block_dim, inner_dim in zip(index_slice, index_block, index_inner):
        assert sub_cumsum[0] == 0, "index_slice 的每一维度 必须是从0开始的"
        sub_cumsum = np.array(sub_cumsum)
        index_combine_dim = sub_cumsum[block_dim] + inner_dim
        index_combine.append(index_combine_dim)
    return np.array(index_combine)


def remap(input: np.ndarray, random=False):
    _input = input.flatten()  # * flatten 这里会产生一次copy
    output = np.zeros_like(_input, dtype=float)
    unique_input = np.sort(np.unique(_input)).tolist()
    remap = np.arange(len(unique_input)) / (len(unique_input) - 1)
    if random:
        remap = np.random.choice(remap, len(remap), replace=False)
    for i, v in enumerate(_input):
        index = unique_input.index(v)
        output[i] = remap[index]
    return output


def from_index(index_x, *args, shape: Sequence[int] = (0,), dtype=float):
    if len(args) != 0:
        index_x = np.array([*index_x, *args])
    if shape != (0,):  # * 如果手动设定了 形状，则要保持一致
        assert len(index_x) == len(shape), "索引要么拆分，要么聚合在第一个参数中"
    else:
        shape = np.max(index_x, axis=1) + 1
    np_output = np.zeros(shape=shape, dtype=dtype)
    np_output[*index_x] = 1
    return np_output


def index_value_2d(input: np.ndarray, nonzero=False):
    """
    有机会检查一下，这个函数的功能是否可以通过下面的代码实现

    .. code-block:: python
        comb = np.indices(input.shape).reshape(2, -1).T
    """
    if nonzero:
        return index_value_2d_nonzero(input)
    else:
        return index_value_2d_(input)


def bisect_right(a: np.ndarray, x, lo=0, hi: Optional[int] = None):  # ? 这里的问题就是，如果输入的是 float，则返回的是 int，但是会被判断为 float，所以有点矛盾
    """
    .. code-block:: python

        np_cumsum = array([ 0, 10, 17, 24])

        np_enhance.bisect_right(np_cumsum, 0)
        >>> 1

        np_enhance.bisect_right(np_cumsum,  1)
        np_enhance.bisect_right(np_cumsum, 15)
        >>> 1
        >>> 2

        np_enhance.bisect_right(np_cumsum, 17)
        >>> 3

    """
    assert a.ndim == 1
    if lo < 0:
        raise ValueError("lo must be non-negative")
    if hi is None:
        hi = len(a)
    if isinstance(x, np.ndarray):
        return cy_bisect_right_array(a, x, lo, hi)
    else:
        return cy_bisect_right(a, x, lo, hi)


def bisect_left(a: np.ndarray, x, lo=0, hi: Optional[int] = None):
    """
    .. code-block:: python

        >>> np_cumsum = array([ 0, 10, 17, 24])

        >>> np_enhance.bisect_left(np_cumsum,  0)
        ... 0

        >>> np_enhance.bisect_left(np_cumsum,   1)
        >>> np_enhance.bisect_left(np_cumsum,  15)
        ... 1
        ... 2

        >>> np_enhance.bisect_left(np_cumsum,  17)
        ... 2

    """
    assert a.ndim == 1
    if lo < 0:
        raise ValueError("lo must be non-negative")
    if hi is None:
        hi = len(a)
    if isinstance(x, np.ndarray):
        return cy_bisect_left_array(a, x, lo, hi)
    else:
        return cy_bisect_left(a, x, lo, hi)


def sparse_lower_set(input_array: sparse.spmatrix, threshold, value):
    """
    优化函数

    important
    ---------

    :code:`SparseEfficiencyWarning: Comparing a sparse matrix with a scalar greater than zero using < is inefficient, try using >= instead.`

    因为大型稀疏矩阵的情况下，大多数元素都为0，在用 :code:`<` 号判断时，很多为0的元素也需要重新赋值，所以这种方式极其不好，可以看到 `warning` 中说到 :code:`>=` 的判断要更有效率。
    那么,我们可以这样来选取，例如不是所有小于 `3` 的元素都置为 `0` ，而是在所有非 `0` 元素中将小于 `3` 的置为 `0`


    Args:
        input_array: 待处理的稀疏矩阵, 稀疏矩阵必须是二维的
        threshold (float|int):  阈值，需要大于 `0` . 小于该阈值时将其置为 **value** ，可以是值也可以是一个矩阵(如果是矩阵的话，其长度应当和索引出来的结果的长度一致)

    """
    assert threshold > 0, "threshold 必须大于0"
    nonzero = input_array.nonzero()  # type: ignore
    nonzero_mask = input_array[nonzero].A[0] < threshold  # type: ignore
    rows = nonzero[0][nonzero_mask]
    cols = nonzero[1][nonzero_mask]
    input_array[rows, cols] = value  # type: ignore
    return input_array


def try_convert_any_to_ndarray(data) -> np.ndarray:
    """
    试图将任何类型转换为 转换为 numpy.ndarray

    当前支持的类型:
        - torch.Tensor
    """
    if str(type(data)) == "<class 'torch.Tensor'>":
        try:
            return data.detach().cpu().numpy()

        except ImportError:
            raise ImportError("torch is not installed, please install it first.")

    raise TypeError(f"未识别的类型: {type(data)}")


def try_convert_any_to_sparray(data) -> sparse.sparray:
    """
    试图将任何类型转换为 scipy.sparse.sparray
    """
    if str(type(data)) == "<class 'torch.Tensor'>":
        try:
            import torch
        except ImportError:
            raise ImportError("torch is not installed, please install it first.")
    raise TypeError(f"未识别的类型: {type(data)}")


class Utils_Sparse:
    @staticmethod
    def format_index_shape(index, shape, m, n) -> Tuple[int, int]:
        if shape is not None:
            m, n = shape
        if m is None:
            m = int(index[0].max())
        if n is None:
            n = int(index[1].max())
        assert n is not None
        assert m is not None
        return m, n

    @classmethod
    def to_scipy_sparse(
        cls,
        index: np.ndarray,
        value: Optional[np.ndarray] = None,
        shape: Optional[Tuple[int, int]] = None,
        m: Optional[int] = None,
        n: Optional[int] = None,
        type_sparse="coo",
    ):
        """
        Args:
            index (torch.Tensor): 稀疏矩阵的索引
            value (torch.Tensor): 稀疏矩阵的值
            m (int): 稀疏矩阵的行数
            n (Optional[int]): 稀疏矩阵的列数
            sparse_type (str): 稀疏矩阵的类型，可选值："coo", "csr", "csc", "bsr", "dia", "lil", "dok"

        """
        from scipy import sparse

        assert type_sparse in ["coo", "csr", "csc", "bsr", "dia", "lil", "dok"]
        assert len(index) == 2

        m, n = cls.format_index_shape(index, shape, m, n)

        if value is None:
            data = np.ones(index.shape[1])
        else:
            data = value

        row, col = index

        assert data.shape[0] == index.shape[1]
        if type_sparse == "coo":
            return sparse.coo_matrix((data, (row, col)), (m, n))
        elif type_sparse == "csr":
            return sparse.csr_matrix((data, (row, col)), (m, n))
        elif type_sparse == "csc":
            return sparse.csc_matrix((data, (row, col)), (m, n))
        elif type_sparse == "bsr":
            return sparse.bsr_matrix((data, (row, col)), (m, n))
        elif type_sparse == "dia":
            return sparse.dia_matrix((data, (row, col)), (m, n))
        elif type_sparse == "lil":
            return sparse.lil_matrix((data, (row, col)), (m, n))
        elif type_sparse == "dok":
            return sparse.dok_matrix((data, (row, col)), (m, n))
        else:
            raise NotImplementedError(f"sparse_type {type_sparse} is not supported")

    @classmethod
    def sum(cls, seq_sparray: Sequence[sparse.spmatrix]) -> sparse.spmatrix:
        """
        基础功能，被 :class:`torch_enhance.Utils_Sparse` 包含
        """
        from copy import deepcopy

        shape = seq_sparray[0].shape
        list_matrix = []
        for sparray in seq_sparray:
            assert sparray.shape == shape
            sparray = sparray.tocsr()  # type: ignore
            list_matrix.append(sparray)

        base_matrix = deepcopy(list_matrix.pop())
        for matrix in list_matrix:
            base_matrix += matrix
        return base_matrix

    @classmethod
    def mean(cls, seq_sparray: Sequence[sparse.spmatrix]) -> sparse.spmatrix:
        """
        基础功能，被 :class:`torch_enhance.Utils_Sparse` 包含
        """
        sum_sparray = cls.sum(seq_sparray)
        sum_sparray *= 1 / len(seq_sparray)
        return sum_sparray


# 计算IoU，矩形框的坐标形式为xyxy
def box_iou_ltrb(box1, box2):
    # 获取box1左上角和右下角的坐标
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    # 计算box1的面积
    s1 = (y1max - y1min + 1.0) * (x1max - x1min + 1.0)
    # 获取box2左上角和右下角的坐标
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    # 计算box2的面积
    s2 = (y2max - y2min + 1.0) * (x2max - x2min + 1.0)

    # 计算相交矩形框的坐标
    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)
    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)
    # 计算相交矩形行的高度、宽度、面积
    inter_h = np.maximum(ymax - ymin + 1.0, 0.0)
    inter_w = np.maximum(xmax - xmin + 1.0, 0.0)
    intersection = inter_h * inter_w
    # 计算相并面积
    union = s1 + s2 - intersection
    # 计算交并比
    iou = intersection / union
    return iou


# 计算IoU，矩形框的坐标形式为xywh
def box_iou_xywh(box1, box2):
    x1min, y1min = box1[0] - box1[2] / 2.0, box1[1] - box1[3] / 2.0
    x1max, y1max = box1[0] + box1[2] / 2.0, box1[1] + box1[3] / 2.0
    s1 = box1[2] * box1[3]

    x2min, y2min = box2[0] - box2[2] / 2.0, box2[1] - box2[3] / 2.0
    x2max, y2max = box2[0] + box2[2] / 2.0, box2[1] + box2[3] / 2.0
    s2 = box2[2] * box2[3]

    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)
    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)
    inter_h = np.maximum(ymax - ymin, 0.0)
    inter_w = np.maximum(xmax - xmin, 0.0)
    intersection = inter_h * inter_w

    union = s1 + s2 - intersection
    iou = intersection / union
    return iou


def split_by_cumsum(np_to_split: np.ndarray, np_cumsum: Union[np.ndarray, list[int]], axis: int = 0) -> list[np.ndarray]:
    """

    >>> x.shape
    ... (21, 2048)
    >>> [i.shape for i in split_by_cumsum(np_to_split, [0, 512, 1024, 2048], axis=1)]
    ... [(21, 512), (21, 512), (21, 1024)]
    """
    assert np_to_split.shape[axis] == np_cumsum[-1]
    list_np = []
    _np_to_split = np_to_split.swapaxes(0, axis)
    for i in range(len(np_cumsum) - 1):
        tmp = _np_to_split[np_cumsum[i] : np_cumsum[i + 1]]
        tmp = tmp.swapaxes(axis, 0)
        list_np.append(tmp)
    return list_np
