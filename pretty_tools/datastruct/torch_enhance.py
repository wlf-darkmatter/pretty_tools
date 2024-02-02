import copy
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy as sp
import torch
import torch.nn.functional as F
from scipy import sparse
from torch import Tensor


def index_to_mask(index: Tensor, size: Optional[int] = None) -> Tensor:
    r"""
    Copy from PyG.utils.mask

    Converts indices to a mask representation.

    Args:
        idx (Tensor): The indices.
        size (int, optional). The size of the mask. If set to :obj:`None`, a
            minimal sized output mask is returned.

    Example:

        >>> index = torch.tensor([1, 3, 5])
        >>> index_to_mask(index)
        tensor([False,  True, False,  True, False,  True])

        >>> index_to_mask(index, size=7)
        tensor([False,  True, False,  True, False,  True, False])
    """
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


def mask_to_index(mask: Tensor) -> Tensor:
    r"""
    Copy from PyG.utils.mask

    Converts a mask to an index representation.

    Args:
        mask (Tensor): The mask.

    Example:

        >>> mask = torch.tensor([False, True, False])
        >>> mask_to_index(mask)
        tensor([1])
    """
    return mask.nonzero(as_tuple=False).view(-1)


def dense_to_sparse(adj: Tensor) -> Tuple[Tensor, Tensor]:
    r"""
    Copy from PyG.utils.sparse

    Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes.

    Args:
        adj (Tensor): The dense adjacency matrix of shape
            :obj:`[num_nodes, num_nodes]` or
            :obj:`[batch_size, num_nodes, num_nodes]`.

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Examples:

        >>> # Forr a single adjacency matrix
        >>> adj = torch.tensor([[3, 1],
        ...                     [2, 0]])
        >>> dense_to_sparse(adj)
        (tensor([[0, 0, 1],
                [0, 1, 0]]),
        tensor([3, 1, 2]))

        >>> # For two adjacency matrixes
        >>> adj = torch.tensor([[[3, 1],
        ...                      [2, 0]],
        ...                     [[0, 1],
        ...                      [0, 2]]])
        >>> dense_to_sparse(adj)
        (tensor([[0, 0, 1, 2, 3],
                [0, 1, 0, 3, 3]]),
        tensor([3, 1, 2, 1, 2]))
    """
    if adj.dim() < 2 or adj.dim() > 3:
        raise ValueError(f"Dense adjacency matrix 'adj' must be 2- or " f"3-dimensional (got {adj.dim()} dimensions)")

    edge_index = adj.nonzero().t()

    if edge_index.size(0) == 2:
        edge_attr = adj[edge_index[0], edge_index[1]]
        return edge_index, edge_attr
    else:
        edge_attr = adj[edge_index[0], edge_index[1], edge_index[2]]
        row = edge_index[1] + adj.size(-2) * edge_index[0]
        col = edge_index[2] + adj.size(-1) * edge_index[0]
        return torch.stack([row, col], dim=0), edge_attr


def block_group_separate(
    node_aff_matrix: Tensor,
    cluster: Tensor,
    method="indice",
    num_block: Optional[int] = None,
) -> tuple[Tensor, Tensor]:
    """
    `group_separate`

    将一个相关性矩阵拆分成两个，分别为自相关和交叉相关

    Args:
        node_aff_matrix (torch.Tensor): 节点相关性矩阵
        cluster (torch.Tensor): 节点所属的簇的索引，或节点所属的簇的累计索引.

    .. note::

        - 如果 **cluster** 是 :code:`indice`, 则将 **cluster** 作为 **cluster_indice** 使用 (**默认**)
        - 如果 **cluster** 是 :code:`cumsum`, 则将 **cluster** 作为 **cluster_cumsum** 使用 (**暂不支持**)


    return:
        tuple[torch.Tensor, torch.Tensor]: 自相关矩阵 :code:`aff_self` 与交叉相关矩阵 :code:`aff_cross`

    todo: 但是应当可以处理 批次化的图结构数据
    """
    assert method in ["indice", "cumsum"]
    assert node_aff_matrix.ndim == 2
    assert node_aff_matrix.shape[0] == node_aff_matrix.shape[1]

    if method == "indice":
        assert cluster.ndim == 1
        assert cluster.shape[0] == node_aff_matrix.shape[0]
        if num_block is None:
            num_block = int(cluster.max().detach().cpu()) + 1
        # * 生成 mesh
        aff_mesh = torch.zeros(node_aff_matrix.shape[:2], dtype=torch.bool, requires_grad=False).to(node_aff_matrix.device)
        for i in range(num_block):
            index_array = cluster == i
            aff_mesh[index_array] = index_array
        aff_self = node_aff_matrix.masked_fill(~aff_mesh, 0)
        aff_cross = node_aff_matrix.masked_fill(aff_mesh, 0)
        return aff_self, aff_cross

    elif method == "cumsum":
        raise NotImplementedError("暂时不支持 cluster_cum 的情况")
    else:
        raise ValueError("必须输入 cluster_indice 或者 cluster_cum")


def torch_quick_visual_2d(data: Tensor, path_save=None):
    """快速可视化一个二维的 :class:`torch.Tensor` 变量

    Args:
        data (torch.Tensor): 输入的二维 :class:`torch.Tensor` 变量

    Returns:
        matplotlib.figure.Figure: 可视化的图像
    """

    import seaborn as sns
    from matplotlib import pyplot as plt

    # Tensor 的 shape 表示为 行数和列数
    # matplotlib 中的 figsize 表示为 宽高
    fig_size = np.array(data.shape).T
    fig = plt.figure(figsize=[*(fig_size * 0.7)], dpi=100)
    assert data.ndim == 2, "只能可视化2维数据"
    data = data.detach().cpu()
    if fig_size.max() <= 100:
        sns.heatmap(data, cmap="Blues", annot=True, fmt=".2f", ax=fig.gca())
    else:
        sns.heatmap(data, cmap="Blues", ax=fig.gca())
    if path_save is not None:
        fig.savefig(path_save)
    return fig


class distance_tools:
    """
    .. note::
        **Stable** 模块，长期支持

    距离度量计算工具, 使用 **torch** 作为运行后端


    特征距离部分，目前支持设置 :code:`metric` 为以下几个:
        - 余弦距离: `cosine`
        - 闵可夫斯基距离: `minkowski`
        - 欧氏距离: `euclidean`


    """

    @staticmethod
    def calc_embedding_distance(data_a: Tensor, data_b: Tensor, metric="cosine", *args, **kwargs) -> Tensor:
        """
        输入两组特征向量，计算特征距离矩阵

        Args:
            data_a (torch.Tensor): :math:`(n, d)` 向量
            data_b (torch.Tensor): :math:`(m, d)` 向量

        .. note::
            ! 输出的距离越 **小** 越相似
        """
        # from scipy.spatial.distance import cdist
        assert data_a.ndim == 2 and data_b.ndim == 2, "输入的数据必须是二维的"
        assert data_a.shape[1] == data_b.shape[1], "输入的深度必须是一致"
        dist_matrix = torch.zeros((len(data_a), len(data_b)), dtype=torch.float)
        if dist_matrix.size == 0:
            return dist_matrix
        if metric == "minkowski":
            dist_matrix = torch.cdist(data_a, data_b, *args, **kwargs)
        elif metric == "cosine":
            dist_matrix = 1 - F.cosine_similarity(data_a.unsqueeze(1), data_b.unsqueeze(0), dim=2)
        elif metric == "euclidean":
            dist_matrix = torch.cdist(data_a, data_b)  #! 注意，使用欧氏距离的时候，这里的距离范围并没有限定在 [0, 1] 之间
        else:
            raise NotImplementedError("不支持的距离度量")

        dist_matrix = torch.maximum(0, dist_matrix)
        return dist_matrix

    @staticmethod
    def calc_embedding_similarity(data_a: Tensor, data_b: Tensor, metric="cosine", *args, **kwargs) -> Tensor:
        """
        输入两组特征向量，计算特征相似度矩阵

        Args:
            data_a (torch.Tensor): :math:`(n, d)` 向量
            data_b (torch.Tensor): :math:`(m, d)` 向量

        .. note::
            ! 输出的距离越 **大** 越相似
        """
        assert data_a.ndim == 2 and data_b.ndim == 2, "输入的数据必须是二维的"
        assert data_a.shape[1] == data_b.shape[1], "输入的深度必须是一致"
        if metric == "cosine":
            sim_matrix = F.cosine_similarity(data_a.unsqueeze(1), data_b.unsqueeze(0), dim=2)
            return sim_matrix
        else:
            dist_matrix = distance_tools.calc_embedding_distance(data_a, data_b, metric, *args, **kwargs)

            return 1 - dist_matrix


from pretty_tools.datastruct import np_enhance


class Utils_Sparse:
    """
    .. note::
        **Stable** 模块，长期支持

    torch_enhance 中的 稀疏矩阵工具类
    """

    @classmethod
    def format_index_shape(cls, index, shape, m, n) -> Tuple[int, int]:
        from pretty_tools.datastruct import np_enhance

        m, n = np_enhance.Utils_Sparse.format_index_shape(index, shape, m, n)
        return m, n

    @classmethod
    def torch_sparse_to_scipy_sparse(cls, adj: torch.Tensor, shape: Optional[Union[tuple, list]] = None):
        assert adj.is_sparse, "输入的矩阵必须是稀疏矩阵"
        layout = str(adj.layout).split("_")[-1]
        row = adj._indices()[0].detach().cpu()
        col = adj._indices()[1].detach().cpu()
        data = adj._values().detach().cpu()
        if shape is None:
            shape = adj.size()
        result = eval(f"sparse.{layout}_matrix((data, (row, col)), shape=shape)")
        return result

    @classmethod
    def to_scipy_sparse(
        cls,
        index: Union[torch.Tensor, np.ndarray],
        value: Optional[Union[torch.Tensor, np.ndarray]] = None,
        shape: Optional[Tuple[int, int]] = None,
        m: Optional[int] = None,
        n: Optional[int] = None,
        type_sparse="coo",
    ):
        """
        格式转换后，通过调用 :func:`np_enhance.Utils_Sparse.to_scipy_sparse` 实现
        Args:
            index (torch.Tensor): 稀疏矩阵的索引
            value (torch.Tensor): 稀疏矩阵的值
            m (int): 稀疏矩阵的行数
            n (Optional[int]): 稀疏矩阵的列数
            sparse_type (str): 稀疏矩阵的类型，可选值："coo", "csr", "csc", "bsr", "dia", "lil", "dok"
        """
        if isinstance(index, torch.Tensor):
            index = index.detach().cpu().numpy()

        if value is None:
            value = np.ones(index.shape[1])
        elif isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        assert isinstance(value, np.ndarray) or value is None
        assert isinstance(index, np.ndarray)
        return np_enhance.Utils_Sparse.to_scipy_sparse(index, value, shape, m, n, type_sparse=type_sparse)

    @classmethod
    def to_torch_sparse(
        cls,
        index: Union[torch.Tensor, np.ndarray],
        value: Optional[Union[torch.Tensor, np.ndarray]] = None,
        shape: Optional[Tuple[int, int]] = None,
        m: Optional[int] = None,
        n: Optional[int] = None,
    ):
        """
        将 **edge_index** 和 **value** 转换成 :class:`torch` 的稀疏矩阵，实际上是调用的 :func:`sparse_coo_tensor` 方法

        Args:
            index (Union[torch.Tensor, np.ndarray]): 稀疏矩阵的索引
            value (Optional[Union[torch.Tensor, np.ndarray]]): 稀疏矩阵的值
            m (Optional[int]): 稀疏矩阵的行数
            n (Optional[int]): 稀疏矩阵的列数
        """
        if shape is None and m is None and n is None:
            print("Warning, to_torch_sparse() doesn't get shape, m and n, which is dangerous.")
        assert len(index) == 2
        if isinstance(index, np.ndarray):
            index = torch.from_numpy(index)
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)

        m, n = cls.format_index_shape(index, shape, m, n)

        if value is None:
            data = torch.ones(index.shape[1]).to(index)
        else:
            data = value

        assert data.shape[0] == index.shape[1]
        return torch.sparse_coo_tensor(index, data, (m, n))

    @staticmethod
    def from_scipy(A):
        """
        Args:
            A (scipy.sparse.coo_matrix): 稀疏矩阵
        """
        A = A.tocoo()
        row, col, value = A.row.astype(np.int64), A.col.astype(np.int64), A.data
        row, col, value = torch.from_numpy(row), torch.from_numpy(col), torch.from_numpy(value)
        index = torch.stack([row, col], dim=0)
        return index, value

    @classmethod
    def sum(cls, seq_sparray: Sequence[Union[sparse.spmatrix, torch.Tensor]]):
        if isinstance(seq_sparray[0], sparse.spmatrix):
            return np_enhance.Utils_Sparse.sum(seq_sparray)  # type: ignore
        else:
            raise NotImplementedError("torch 的 sum 方法还没有补充")

    @classmethod
    def mean(cls, seq_sparray: Sequence[Union[sparse.spmatrix, torch.Tensor]]):
        sum_sparray = cls.sum(seq_sparray)
        sum_sparray *= 1 / len(seq_sparray)
        return sum_sparray
