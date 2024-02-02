r"""
全局定义
---------

* 节点个数: numbers of nodes
	.. math::	n=|\mathcal V|
* 边个数: number of edges
	.. math::	m=|\mathcal E|
* 节点: node
	.. math::	\mathcal v_i \in \mathcal V
	.. math::	\mathcal V = \{v_1,v_2,\dots,v_n\}
* 节点特征: d-dimensional feature of nodes
	.. math::	\pmb X \in \mathbb R^{n\times d_v}
* 隐含层节点特征
	.. math::	\pmb h_i
* 边: edge
	.. math::	e_{i,j} \in \mathcal E
* 边特征: d-dimensional feature of edges
	.. math::	\pmb E \in \mathbb R^{m \times d_e}
* 节点维度数: edge feature dimension
	.. math::	d_v
* 边特征维度数: edge feature dimension
	.. math::	d_e
* 邻接矩阵: adjacency matrix (通常是一个 binary matrix)
	.. math::	\pmb A \in \{0,1\}^{n \times n}
* 度: degree
	节点 :math:`i` 的度————该节点的邻接节点个数

	.. math::	k_i = \sum_{j \in \mathcal E} A(i, j)

异构图定义
-------------

异构图比较复杂，

.. math::
    \mathcal G=(\mathcal V,\mathcal E,\tau,\phi)

.. image:: http://pb.x-contion.top/wiki/2023_07/31/3_%E5%BC%82%E6%9E%84%E5%9B%BE.png
    :alt: 异构图示意图
    :height: 900px
    :width: 1382px
    :scale: 30%

- 节点： :math:`v \in \mathcal V`
- 节点类型： :math:`\tau(v)`
- 边： :math:`(u, v) \in \mathcal E`
- 边类型： :math:`\phi(u, v)`
- 完整的(点-边-点)类型元组： :math:`r(u,v)=(\tau(u), \phi(u,v), \tau(v))`


------------------

* 图结构: network, graph
	.. math::	\mathcal G=(\mathcal V, \mathcal E)

* 幂次图:
	.. math::	\mathcal G^k=(\mathcal V,\mathcal E^k)

	对于 :math:`(u,v) \in\mathcal E^k` ， :math:`\mathcal G` 中一定存在一条最多由 :math:`k` 条边构成的从 :math:`u` 到 :math:`v` 的路径
* 复合图: composite graph
	.. math::	\mathcal G_{com}=\left(\sum{\mathcal V^{(i)}},\sum{\mathcal E^{(i)}}\right)

* 复合邻接矩阵
	.. math::	\pmb A_{com}=\left[ \begin{matrix}\pmb A^{\left( 1\right) }&0&\cdots &0\\ 0&\pmb A^{\left( 2\right) }&\cdots &0\\ \vdots &\vdots &\ddots &\vdots \\ 0&0&\cdots &\pmb A^{\left( n\right) }\end{matrix} \right]

* 复合节点特征
	.. math::	\pmb X_{com}=\left[ \pmb X^{\left( 1\right)  },\pmb X^{\left( 2\right)  },\cdots ,\pmb X^{\left( n\right)  }\right]
* 复合节点边特征
	.. math::	\pmb E_{com}=\left[ \pmb E^{\left( 1\right)  },\pmb E^{\left( 2\right)  },\cdots ,\pmb E^{\left( n\right)  }\right]
* 真值: ground truth
    真值是复合图中，新创建的邻接矩阵关系，用于表示两个相机中的目标是否是同一个目标，输出的目标相似度中

    .. math::	\pmb Y \in \mathbb R^{N \times d_y}


多相机多目标图匹配数学定义
--------------------------

* 多相机复合图结构: Multi-Camera Cohering Graph
	.. math::	\mathbb G=\left(\mathcal G_1,\mathcal G_2,\dots,\mathcal G_C\right)

    多相机复合图结构是一种高阶的复合图，这里 :math:`C` 定义为同一个场景下的相机个数。

    其中，每个相机中的节点个数为 :math:`|\mathcal G_i|=n_i`




"""
import warnings
from copy import deepcopy
from functools import lru_cache
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import scipy.sparse as sci_sparse
import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
import torch_sparse
from scipy.sparse import bsr_matrix, coo_matrix, csc_array, csr_array, lil_array, sparray, spmatrix
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.data.collate import collate
from torch_geometric.loader.dataloader import Collater
from torch_geometric.typing import OptTensor

T = TypeVar("T")
Array_Like = TypeVar("Array_Like", torch.Tensor, np.ndarray)


class CohereGraph:
    """
    重新思考这个类的设计，把一些不必要的功能给舍弃掉

    这里最核心的就是一个 全相机 graph，不要添加其他没有必要的功能，导致性能不好

    :note: 由于关系到赋值问题，所以这里并不将 :code:`self.data` 的属性暴露出来进行封装，而是直接操作 :code:`self.data` 的属性即可

    Attributes:
        backend (str): 用于指定使用的后端，可选值为 "torch" 或者 "numpy"，默认为 "numpy",

    """

    backend = "numpy"

    def __init__(
        self,
        x,
        edge_index=None,
        list_len: Optional[Sequence[int]] = None,
        x_pos=None,
        dtype=None,
        device=None,
        y=None,
    ) -> None:
        """

        args:
            x: :code:`shape=[n, d]` 这里的x是所有节点的特征
            edge_index: :code:`shape=[2, m]` 这里的 edge_index 应当就是拼接处理后的
            list_len: 每个相机的节点个数, **这里并不是 cumsum 累加求和后的**
            x_pos: :code:`shape=[n, 2]` 节点的位置信息，位置信息应当是根据图像尺寸进行了归一化操作的，分布在 :code`[0, 1]` 之间


        .. note::

            如果是其他格式，那也是先生成了numpy，然后转换过去的

            内部的 self.data.x 的顺序一定是按照相机的顺序排列的

        """
        warnings.warn(
            "This class is deprecated and will be removed in the future.",
            DeprecationWarning,
        )
        self.data = Data(x, edge_index)

        if device is None:
            self.device = "cpu"
        else:
            self.device = device

        self._dtype = None
        if dtype is not None:
            self.dtype = dtype

        if list_len is not None:
            self.list_len = list_len
        else:
            self.list_len = [len(x)]  # * 如果没有输入，则将整个视为一整块进行处理

        if x_pos is not None:
            assert x_pos.shape[0] == x.shape[0], "x_pos.shape[0] != x.shape[0], 两者应当具有相同的节点个数"
            assert x_pos.shape[1] == 2, "x_pos.shape[1] != 2, 位置信息应当是二维的, shape 的定义和 :code:`edge_index` 不一致"
            self.data["pos"] = x_pos

        if y is not None:
            assert y.shape[0] == x.shape[0], "y.shape[0] != x.shape[0], 两者应当具有相同的节点个数"
            self.data["y"] = y

        self.to(device)

    def to(self, device):
        self.device = device
        self.data.to(device)

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        """
        # todo 这里之后可能要区分一下 numpy 和 torch
        """
        self._dtype = dtype
        if self.data.x is not None:
            self.data["x"] = self.data.x.type(dtype)
        if self.data.edge_index is not None:
            self.data["edge_index"] = self.data.edge_index.type(dtype)
        if self.data.pos is not None:
            self.data["pos"] = self.data.pos.type(dtype)
        if self.data.edge_attr is not None:
            self.data["edge_attr"] = self.data.edge_attr.type(dtype)
        if self.data.y is not None:
            self.data["y"] = self.data.y.type(dtype)

    @property
    def num_block(self) -> int:
        """
        相机个数，也可以说是 **分块** (`block`) 的个数
        """
        return len(self.list_len)

    @property
    @lru_cache(maxsize=1)
    def cumsum(self) -> np.ndarray:
        """
        获取每个相机的节点个数的累加和，起始项为零
        """
        return np.cumsum([0] + self.list_len)  # type: ignore

    @property
    def num_nodes(self) -> int:
        if self.data.num_nodes is None:
            return 0
        else:
            return self.data.num_nodes

    @property
    def x_onehot(self):
        """
        用以区分不同 **分块**(`block`) 的 one-hot 编码, 编码数从 :code:`0` 开始
        """
        from pretty_tools.datastruct.np_enhance import cy_onehot_by_cumulate

        result = cy_onehot_by_cumulate(self.cumsum, 0)

        if self.backend == "numpy":
            return result

        elif self.backend == "torch":
            import torch

            return torch.as_tensor(result).to(self.device)  # as_tensor 会共享内存，反正这里不需要复制这个矩阵，就用 as_tensor 吧

    @property
    @lru_cache(maxsize=1)
    def x_cluster(self):
        """
        输出一个数组，用以区分不同 **分块** (`block`) 的索引, 索引数从 :code:`0` 开始

        example
        -------
        .. code::

            a: CohereGraph
            a.cumsum
            >>> array([ 0, 10, 17, 24])

            a.x_cluster
            >>> array([
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                1., 1., 1., 1., 1., 1., 1.,
                2., 2., 2., 2., 2., 2., 2.])

        """
        from pretty_tools.datastruct.np_enhance import cy_cum_to_index

        output = cy_cum_to_index(self.cumsum)
        if self.backend == "numpy":
            return output
        else:
            import torch

            return torch.as_tensor(output).to(self.device)  # as_tensor 会共享内存，反正这里不需要复制这个矩阵，就用 as_tensor 吧

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_block={len(self.list_len)}, list_len={self.list_len})"

    @classmethod
    def merge_from(cls, list_graph, dtype=None, device=None):
        """
        将多个图结构合并组成一个大的图结构，同时设定一个
        """
        import torch

        list_len = [len(graph.x) for graph in list_graph]

        new_x = torch.cat([graph.x for graph in list_graph])
        last_len = 0
        new_edge_index = []
        combine_pos = []
        for graph in list_graph:
            if graph.edge_index is not None:  #! 巨大傻逼的 bug，这里的 edge_index 有可能是 None（因为只有一个节点）
                new_edge_index.append(graph.edge_index + last_len)
            last_len += len(graph.x)
            combine_pos.append(torch.Tensor(graph.pos[:, :2]))
        new_edge_index = torch.cat(new_edge_index, dim=1)
        combine_pos = torch.cat(combine_pos, dim=0)
        result = cls(
            new_x,
            new_edge_index,
            list_len=list_len,
            x_pos=combine_pos,
            dtype=dtype,
            device=device,
        )

        return result


def block_pair_separate(
    node_aff_matrix: torch.Tensor,
    cluster: torch.Tensor,
    method="indice",
    num_block: Optional[int] = None,
) -> np.ndarray:
    """pair_separate

    将一个相关性矩阵拆分成两个，分别为自相关和交叉相关
    目前假设相关度矩阵是二维的，后续要支持三维，多出来的维度指的是相关度特征，即 `(n, m, d)`

    Args:
        node_aff_matrix (torch.Tensor): 节点相关性矩阵
        cluster (torch.Tensor): 节点所属的簇的索引，或节点所属的簇的累计索引.

    .. note::

        - 如果 :code:`cluster` 是 :code:`indice`, 则将 :code:`cluster` 作为 **cluster_indice** 使用 (**默认**)
        - 如果 :code:`cluster` 是 :code:`cumsum`, 则将 :code:`cluster` 作为 **cluster_cumsum** 使用 (**暂不支持**)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: _description_
    """
    assert method in ["indice", "cumsum"]
    assert node_aff_matrix.ndim == 2, "目前假设相关度矩阵是二维的，后续要支持三维，多出来的维度指的是相关度特征，即 `(n, m, d)`"
    if method == "indice":
        if num_block is None:
            num_block = int(cluster.max().detach().cpu()) + 1
        torch_result = np.empty((num_block, num_block), dtype=np.object_)

        for i in range(num_block):
            torch_array_i = node_aff_matrix[cluster == i]
            for j in range(num_block):
                torch_result[i, j] = torch_array_i[:, cluster == j]
        return torch_result
    elif method == "cumsum":
        raise NotImplementedError("暂时不支持 cluster_cum 的情况")
    else:
        raise ValueError("必须输入 cluster_indice 或者 cluster_cum")


def get_block_pair(y_id: Tensor) -> Tensor:
    """
    输入合成 block 的 id 号，返回各个分块的匹配稀疏矩阵

    Args:
        y_id (torch.Tensor): 合成 block 的 场景全局 `id` 号


    .. code-block:: python

        y_id = torch.Tensor([ 0.,  2.,  3.,  7.,  8., 13., 15., 17., 18., 25.,  0.,  2.,  3.,  5.,
        17., 21., 33.,  2.,  3., 13., 17., 21., 33., 39.])
        get_block_pair(y_id, x_cluster)

    结果可视化

    .. image:: http://pb.x-contion.top/wiki/2023_08/08/3_get_block_pair.png
        :alt: get_block_pair
        :width: 200px
        :height: 200px
    """

    tensor_pair = y_id.unsqueeze(1).expand(len(y_id), len(y_id))
    result_pair = tensor_pair == y_id

    return result_pair


class Batch_Multi_Graph:
    """
    .. note::
        **Stable** 模块，逐步吸收到 XBGraph 中

    多个 **单相机子图** 合并成一个 较大的图，成为 **多相机图**，
    但还要设计成一个可组合的形式，用树的形式可以更好地理解这个数据的结构


    - 一个 **批次化数据** 由多个 **多相机图** 构成
    - 一个 **多相机图** 由多个 **单相机图** 构成
    - 一个 **单相机图** 由多个 **节点** 和 **边** 构成

    .. mermaid::

        graph LR
            Batch_Block ==> Block_Data0
            Batch_Block ==> Block_Data1
            Batch_Block ==> Block_Data2
            Batch_Block ==> Block_Datab

            Block_Data1 --> Data1,0
            Block_Data1 --> Data1,1
            Block_Data1 --> Data1,c

            Data1,1 --- x
            Data1,1 --- edge_index

    .. note::

        :class:`Block_Data` 并没有定义，本质上就是一个 :class:`Data` 类，但是调用了 **PyG** 内部的 batch 方法

        :class:`Batch_Block` 并没有定义，本质上就是一个 :class:`Data` 类，但是调用了 **PyG** 内部的 batch 方法，对 :class:`Block_Data` 进行了再次合并

        尽量不要使用到 batch 属性，这个属性还不确定在重组 batch 的时候会不会发生变化
    """

    keyname = ["cluster", "batch"]
    collect_fn: Callable = Collater(None, None)

    @staticmethod
    def compair_keyname(data0: Union[Data, Batch], data1: Union[Data, Batch]):
        from pretty_tools.echo import X_Table

        table = X_Table(title="Batch_Multi_Graph compare", highlight=True)

        table.add_column("keyname", justify="left")
        table.add_column("data0", justify="left")
        table.add_column("data1", justify="left")
        table.add_column("status", justify="left")

        pass
        for key in Batch_Multi_Graph.keyname:
            if key in data0 and key in data1:
                table.add_row(str(key), str(data0[key].shape), str(data1[key].shape), str())
            elif key in data0 and key not in data1:
                table.add_row(str(key), "None", str(data1[key].shape), "N/A")
            elif not key in data0 and key in data1:
                table.add_row(str(key), str(data0[key].shape), "None", "N/A")
            else:
                table.add_row(str(key), "None", "None", "None")

            pass
        pass
        table.print()

    @classmethod
    def is_BlockData(cls, data: Union[Data, Batch]):
        """
        判断是否为 封装成 :class:`Block_Data` 或者 :class:`Batch_Block` 的样子
        """
        return "len_graph" in data

    @classmethod
    def check_none(cls, list_data: list) -> Tuple[list[int], list[int]]:
        """查看列表中的元素是否有 None，以及是否全部都是 None 的列表索引号

        Args:
            list_data (List[Data]): _description_

        Returns:
            list[int], list[int]
            第一个表示 anyNone ，第二个表示 allNone

        Example:
            list_None, list_notNone = check_none(list_data)
        """

        list_None = []
        list_notNone = []

        for i, data in enumerate(list_data):
            if data is None:
                list_None.append(i)
            else:
                list_notNone.append(i)
        return list_None, list_notNone

    @classmethod
    def Batch_Block_recombine(cls, data: Union[Data, Batch], list_indexs: list[int]):
        """
        重组子图的组合顺序，也能改变组合的大小

        需要注意的是，由于是通过 :func:`Batch.from_data_list` 函数进行的重组，会使得原来子图中的 `batch` 信息丢失 中的数据不变.

        不建议之后继续使用 `batch` 功能。

        """
        assert cls.is_BlockData(data)
        list_batch = data.to_data_list()
        new_list = [list_batch[i] for i in list_indexs]
        result = cls.collect_fn(new_list)

        return result

    @classmethod
    def Batch_Block_merge_from(cls, list_block_data: Sequence[Data]):
        """
        合并多个子图

        Args:
            list_block_data (Sequence[Data]): 子图列表

        .. note::
            这里输入的 :class:`Data` 必须是一个被 :func:`Block_Data_merge_from` 处理好的 Data,

        """
        print("Batch_Block_merge_from 弃用")
        for data in list_block_data:
            assert cls.is_BlockData(data)
        #! 过滤掉 None
        batch = Batch.from_data_list([i for i in list_block_data if i is not None])

        batch["block_ptr"] = np.cumsum(np.concatenate(batch["len_graph"]))
        batch["block_ptr"] = np.insert(batch["block_ptr"], 0, 0)
        batch["batch_ptr"] = np.cumsum([0] + [len(i) for i in batch["len_graph"]])
        batch["batchblock_ptr"] = np.cumsum([0] + [sum(i) for i in batch["len_graph"]])

        return batch

    @classmethod
    def Batch_Block_to_ListBlockData(cls, data: Union[Data, Batch]):
        assert cls.is_BlockData(data)
        ls = data.to_data_list()
        return ls

    @classmethod
    def Batch_adj_to_Listadj(
        cls,
        batch_adj: torch.Tensor,
        batchblock_ptr: np.ndarray,
    ):
        """
        输入的是一个合成的邻接矩阵

        将一个 batch 化的转换为 BlockData 的 adj 列表，输出的也是邻接矩阵，保留权重，返回的是稠密矩阵

        .. note::
            主要还是因为 torch_spare 不支持切片，所以要调用 torch_sparse.SparseTensor 来实现切片操作
        """

        n_block = len(batchblock_ptr) - 1
        assert batch_adj.is_sparse
        batch_adj = batch_adj.coalesce()
        edge_index = batch_adj.indices()

        adj = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], value=batch_adj.values())
        list_adj = []
        for i in range(n_block):
            x1 = batchblock_ptr[i]
            x2 = batchblock_ptr[i + 1]
            list_adj.append(adj[x1:x2, x1:x2].to_dense())  # type: ignore
        return list_adj

    @classmethod
    def Block_Data_merge_from(cls, list_data: list[Data]):
        """
        合并多个子图，被这个工具处理过的子图

        该函数合并完毕后就已经能够进行 batch 化合并了，不一定需要 :func:`Batch_Block_merge_from` 再处理，可以直接被 loader 加载

        区分边界要用 block_ptr 和 batch_ptr， 不能用 cluster，因为可能存在某几个相机没有目标，导致前一个 block 的相机信息和后一个 block 的相机信息缝合了

        Args:
            list_data (list[Data]): 子图列表


        Examples:

            .. code-block:: python

                list_node_x: list[torch.Tensor]
                list_new_graph = []
                for x in list_node_x:
                    list_new_graph.append(Data(x=x))

                block_data = Batch_Multi_Graph.Block_Data_merge_from(list_new_graph)


        .. note::
            :code:`block_data.batch` 其实就是 节点所在的相机图 的索引号，但是多个 :code:`block_data` 合并后，
            这个向量中的索引号会累加，在单独处理的时候会不方便，这里给一个不会累加的信息量

        """
        print("Block_Data_merge_from 弃用")
        # ! 由于可能存在 None，所以当发现存在 None 的时候，要进行数据维度的统计，以方便进行合成

        # 对于一个 batch，空的 block 可以跳过，但是对于一个 block，空的 graph 不可以跳过，因为相机本身是存在着的
        ln, lnn = cls.check_none(list_data)
        if len(lnn) == 0:
            return None

        if len(ln) != 0:
            if len(lnn) != 0:
                notNoneData = list_data[lnn[0]]
                # * 构造空数据
                empty_Data = Data(
                    x=torch.empty((0, notNoneData.x.shape[1])) if "x" in notNoneData else None,
                    y=(torch.empty((0,)) if notNoneData.y.ndim == 1 else torch.empty((0, notNoneData.y.shape[1]))) if "y" in notNoneData else None,
                    pos=torch.empty((0, notNoneData.pos.shape[1])) if "pos" in notNoneData else None,
                    edge_attr=torch.empty((0, notNoneData.edge_attr.shape[1])) if "edge_attr" in notNoneData else None,
                    edge_index=torch.empty((2, 0), dtype=torch.long) if "edge_index" in notNoneData else None,
                )
            for i in ln:
                list_data[i] = deepcopy(empty_Data)

        block_data = Batch.from_data_list([*list_data])

        return block_data

    @classmethod
    def apply_edge_weight(cls, batch_data: T, edge_weight: torch.Tensor) -> T:
        """
        把生成的边权重信息适配到 batch 化的数据上

        Args:
            batch_data (Union[Data, Batch]): 操作的batch数据
            edge_weight (torch.Tensor): 边的权重，要求长度和 :code:`batch_data.edge_index` 边个数一致
        """
        assert edge_weight.device == batch_data.graph["edge_index"].device, f"变量需要在同一个 device 下"  # type: ignore
        batch_data["edge_weight"] = edge_weight  # type: ignore
        assert batch_data.num_edges == edge_weight.shape[0]  # type: ignore
        batch_data.graph._slice_dict["edge_weight"] = batch_data.graph._slice_dict["edge_index"]  # type: ignore
        batch_data.graph._inc_dict["edge_weight"] = batch_data.graph._inc_dict["x"]  # type: ignore
        return batch_data
        # ? debug
        adj_debug = sci_sparse.coo_matrix((np.ones(batch_data.num_edges), batch_data.edge_index.cpu().numpy())).A

    @classmethod
    def apply_edge_attr(cls, batch_data: T, edge_attr: torch.Tensor) -> T:
        """
        把生成的边特征信息适配到 batch 化的数据上

        Args:
            batch_data (Union[Data, Batch]): 操作的batch数据
            edge_attr (torch.Tensor): 边的权重，要求长度和 :code:`batch_data.edge_index` 边个数一致
        """
        assert edge_attr.device == batch_data.graph["edge_index"].device, f"变量需要在同一个 device 下"  # type: ignore
        batch_data["edge_attr"] = edge_attr  # type: ignore
        assert batch_data.num_edges == edge_attr.shape[0]  # type: ignore
        batch_data.graph._slice_dict["edge_attr"] = batch_data.graph._slice_dict["edge_index"]  # type: ignore
        batch_data.graph._inc_dict["edge_attr"] = batch_data.graph._inc_dict["x"]  # type: ignore
        return batch_data

    @classmethod
    def apply_edge_custom(cls, batch_data: T, edge_custom: torch.Tensor, name: str, inc=False) -> T:
        """
        把自定义的属性适配到 batch 化的数据上

        Args:
            batch_data (Union[Data, Batch]): 操作的batch数据
            edge_custom (torch.Tensor): 新添加的属性，要求长度和 :code:`batch_data.edge_index` 边个数一致
            name (str): 自定义的属性名
            inc (bool, optional): 切片时是否和 edge_index 一样，根据产生的偏移进行还原，默认 False
        """
        assert edge_custom.device == batch_data["edge_index"].device, f"变量需要在同一个 device 下"  # type: ignore
        batch_data[name] = edge_custom  # type: ignore
        assert batch_data.num_edges == edge_custom.shape[0]  # type: ignore
        batch_data._slice_dict[name] = batch_data._slice_dict["edge_index"]  # type: ignore
        if inc:
            batch_data._inc_dict[name] = batch_data._inc_dict["edge_index"]  # type: ignore
        else:
            batch_data._inc_dict[name] = batch_data._inc_dict["x"]  # type: ignore
        return batch_data

    @classmethod
    def Batch_Block_extract_edge(cls, batch_data: Data, graph_ptr: np.ndarray, only_self=False, only_cross=False) -> tuple[OptTensor, OptTensor]:
        """
        对一个 Batch_Block 的数据进行处理，

        Args:
            batch_data (Union[Data, Batch]): 操作的batch数据
            only_self (bool, optional): 是否只提取自身的边，默认为False
            only_cross (bool, optional): 是否只提取交叉的边，默认为False

        .. note::

            这里处理大规模数据的时候可能会有点慢，因为使用的是 scipy 系数矩阵的计算方式，没办法放到 GPU 上计算，因为 torch 没有相关的实现，之后有需要再做优化

        """
        n_batch = batch_data.num_graphs
        num_nodes = batch_data.num_nodes
        num_edges = batch_data.num_edges
        # 创建一个 mask

        mask_self = lil_array((num_nodes, num_nodes), dtype=bool)
        for i in range(len(graph_ptr) - 1):  # * 这里发现了巨大 bug，之前用的是 num_batch， 但是这是错误的长度，造成只提取了部分的边
            i_l = graph_ptr[i]
            i_r = graph_ptr[i + 1]
            mask_self[i_l:i_r, i_l:i_r] = True
        # debug_mask_self = mask_self.toarray() #? debug
        adj = csr_array(
            (
                np.ones(num_edges),
                (
                    batch_data["edge_index"][0].detach().cpu().numpy(),
                    batch_data["edge_index"][1].detach().cpu().numpy(),
                ),
            ),
            shape=(num_nodes, num_nodes),
        )
        # debug_adj = adj.toarray() #? debug
        self_edge_index = None
        cross_edge_index = None
        if only_self:
            adj_self = adj * mask_self
            # debug_adj_self = adj_self.toarray() #? debug
            self_edge_index = torch.from_numpy(np.stack(adj_self.nonzero(), axis=0)).to(batch_data["edge_index"])
            return self_edge_index, None

        if only_cross:
            adj_cross = adj
            adj_cross[mask_self] = 0
            # debug_adj_cross = adj_cross.toarray() #? debug
            cross_edge_index = torch.from_numpy(np.stack(adj_cross.nonzero(), axis=0)).to(batch_data["edge_index"])
            return None, cross_edge_index

        else:
            adj_self = adj * mask_self
            adj_cross = adj
            adj_cross[mask_self] = 0
            self_edge_index = torch.from_numpy(np.stack(adj_self.nonzero(), axis=0)).to(batch_data["edge_index"])
            cross_edge_index = torch.from_numpy(np.stack(adj_cross.nonzero(), axis=0)).to(batch_data["edge_index"])

            return self_edge_index, cross_edge_index
        # ? debug
        list_self_edge = []
        for edge in zip(*batch_data["edge_index"].cpu().numpy()):
            if edge[0] == edge[1]:
                list_self_edge.append(edge)

        list_self_edge = []
        for edge in zip(*mask_self.nonzero()):
            if edge[0] == edge[1]:
                list_self_edge.append(edge)

    @classmethod
    def Batch_Block_edge_rebuild(
        cls,
        batch_block: Union[Data, Batch],
        edge_index: OptTensor = None,
        *,
        edge_attr: OptTensor = None,
        edge_weight: OptTensor = None,
        copy: bool = False,
        device=None,
    ):
        """
        重新构建 Batch_Block 单个 batch 中的边关系。
        重新构建一个新的 Batch_Block，内部数据可以是复制的，也可以是共享的。

        :param batch_data:
        :return:

        .. note::
            这里必须确认 节点数量不变

        """
        # adj = pyg_utils.to_dense_adj(edge_index)  #? debug
        # assert cls.is_BlockData(batch_block)
        num_nodes = int(batch_block.num_nodes)  # type: ignore
        batch_size = int(batch_block.num_graphs)
        if edge_index is not None:
            assert len(edge_index) == 2, "edge_index size must be equal to 2"
        if edge_attr is not None:
            if edge_index is not None:
                assert len(edge_attr) == edge_index.shape[1], "edge_attr size must be equal to new num_edges"
            else:
                assert len(edge_attr) == batch_block.edge_index.shape[1], "edge_attr size must be equal to num_nodes"  # type: ignore
        if device is None:
            if batch_block.is_cuda:  # type: ignore
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        list_block: list[Union[BaseData, Data, Batch]] = batch_block.to_data_list()  # type: ignore

        ptr = np.cumsum([0] + [list_block[i].num_nodes for i in range(batch_size)])  # type: ignore
        # * 处理新的边关系
        if edge_index is not None:
            new_adj = Utils_Edge.edgeindex_to_scipy_sparse(edge_index).tocsr()
            for i in range(batch_size):
                i_edge_index: sci_sparse.csr_array = new_adj[ptr[i] : ptr[i + 1], ptr[i] : ptr[i + 1]]  # type: ignore
                list_block[i]["edge_index"] = torch.from_numpy(np.stack(i_edge_index.nonzero())).to(device).to(torch.long)  # type: ignore
        new_batch = cls.Batch_Block_merge_from(list_block)  # type: ignore

        if edge_attr is not None:
            cls.apply_edge_attr(new_batch, edge_attr)  # type: ignore
        if edge_weight is not None:
            cls.apply_edge_weight(new_batch, edge_weight)  # type: ignore
        return new_batch

    @classmethod
    def lengraph_to_blockptr(cls, len_graph):
        block_ptr = np.cumsum(len_graph)
        return np.insert(block_ptr, 0, 0)

    @classmethod
    def Batch_Block_gt_edge_index(
        cls,
        y: Array_Like,
        len_graph: list[np.ndarray],
    ) -> Array_Like:
        """

        如果输入的类型是 ``torch.Tensor``，则返回的也是 ``torch.Tensor``
        """
        if isinstance(y, torch.Tensor):
            _y = y.detach().cpu().numpy()
        else:
            _y = y

        if isinstance(len_graph, np.ndarray):
            len_graph = [len_graph]

        list_gt_edge = []

        offset = 0
        for b in len_graph:
            # * 拆分成 block 进行处理
            block_y = _y[offset : offset + sum(b)]
            # * 应当多带一个用于标记末尾
            gt_edge = cls.Block_Data_gt_edge_index(block_y, np.cumsum([0] + b.tolist()))
            if len(gt_edge) != 0:
                list_gt_edge.append(gt_edge)
            offset += sum(b)

        if len(list_gt_edge) == 0:
            np_batch_gt = np.empty((2, 0))
        else:
            np_batch_gt = np.concatenate(list_gt_edge, axis=1)

        # from pretty_tools.visualization.draw import Pretty_Draw, Visiual_Tools  #? debug
        # vis_gt = Visiual_Tools.fig_to_image(Pretty_Draw.draw_edge_index(np_batch_gt, shape=(len(y), len(y))))  #? debug

        if isinstance(y, torch.Tensor):
            return torch.from_numpy(np_batch_gt).to(y.device)
        else:
            return np_batch_gt

    @classmethod
    def Block_Data_gt_edge_index(
        cls,
        y: Union[np.ndarray, torch.Tensor],
        graph_ptr: np.ndarray,
    ) -> np.ndarray:
        """
        获取当前 Block 的真值匹配
        Args:
            y (Union[np.ndarray, torch.Tensor]): 真值
            graph_ptr (np.ndarray): 分块的索引，

        .. note::
            注意， ``graph_ptr`` 可以有偏移也可以没有偏移量，如果加入的 ``graph_ptr`` 有偏移量，则输出也是有偏移量的，如果加入的是没有偏移量的，则输出也没有偏移量

        """
        import itertools

        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        new_y = np.stack([np.arange(len(y)), y])
        # * 将真值 id 放到一起，进行排序
        sort_new_y = new_y[:, np.lexsort((np.arange(len(y)), y))]  # [n, 2] , 第一个元素是 id，第二个元素是 id 的索引

        y_cluster = sort_new_y[1]
        ptr = np.where(y_cluster[:-1] != y_cluster[1:])[0] + 1
        ptr = np.insert(ptr, 0, 0)
        ptr = np.insert(ptr, len(ptr), len(y_cluster))

        list_comb = []
        for i, n in enumerate(np.diff(ptr)):
            if n != 1:
                np_same_id = sort_new_y[0, ptr[i] : ptr[i + 1]]
                # tmp_a, tmp_b = ptr[i], ptr[i + 1]
                # print(f"sort_new_y[{tmp_a}:{tmp_b}] = {sort_new_y[0,tmp_a:tmp_b]}")
                list_comb += [*itertools.combinations(np_same_id, 2)]
            pass
        match_tri_u = np.array(list_comb).T + graph_ptr[0]
        # from pretty_tools.visualization.draw import Pretty_Draw, Visiual_Tools  #? debug
        # vis_block_gt = Visiual_Tools.fig_to_image(Pretty_Draw.draw_edge_index(match_tri_u, shape=(len(y), len(y)), np_cumsum=block_ptr, Oij=(block_ptr[0], block_ptr[0])))  #? debug
        return match_tri_u

    @classmethod
    def Block_Data_get_cross_slice(
        cls,
        len_graph: Union[np.ndarray, Sequence[int]],
        mothod: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import itertools

        graph_ptr = np.cumsum(len_graph)
        graph_ptr = np.insert(graph_ptr, 0, 0)

        index_black = np.stack([graph_ptr[:-1], graph_ptr[1:]]).T
        n_graph = len(len_graph)
        if mothod == "permutations":
            block_pair = np.array([*itertools.permutations(range(n_graph), 2)])
        if mothod == "combinations":
            block_pair = np.array([*itertools.combinations(range(n_graph), 2)])

        iter_offset = index_black[np.concatenate(block_pair)].reshape(-1, 4)
        return iter_offset, block_pair, np.array([0] + [len(block_pair)])

    @classmethod
    def Batch_Block_get_cross_slice(
        cls,
        list_len_graph: list[np.ndarray],
        mothod: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """

        这个获取的是一个 **Batch_Block** 中所有 Block_Data 区域中所有交叉矩阵的切片位置，每一行对应的是 ``[i0, i1, j0, j1]``

        mothod = permutations 获取的是全部的区域(默认)

        mothod = combinations 获取的是上半三角交叉的区域


        Args:
            mothod (str) : ['permutations', 'combinations']
        Example:


        .. code-block::

            for offset in iter_offset:
                tmp_adj_ij = adj_score[offset[0]:offset[1], offset[2]:offset[3]]


        返回 ``iter_offset`` 和 ``np_block_pair`` 还有 ``np_batch``

        ``np_block_pair`` 表示 ``iter_offset`` 中元素属于哪几个的交叉块

        ``np_batch`` 表示 ``iter_offset`` 中元素属于哪个 **batch**
        """
        import itertools

        graph_ptr = np.cumsum(np.concatenate([[0]] + list_len_graph, axis=0))

        if mothod is None:
            mothod = "permutations"
        assert mothod in ["permutations", "combinations"]

        index_black = np.stack([graph_ptr[:-1], graph_ptr[1:]]).T
        comb = []
        block_pair = []
        slice_batch_ptr = [0]
        offset = 0
        for i, b in enumerate(list_len_graph):
            # b = batch_ptr[i + 1] - offset  # b 是该 block 中的 graph 个数

            n_graph_i = len(b)
            if mothod == "permutations":
                np_comb_i = np.array([*itertools.permutations(range(n_graph_i), 2)])
            if mothod == "combinations":
                np_comb_i = np.array([*itertools.combinations(range(n_graph_i), 2)])

            block_pair.append(np_comb_i)
            # 当前的 len(np_comb_i) 个组合，都是属于第 i 个 batch 的
            slice_batch_ptr += [len(np_comb_i)]
            comb.append(offset + np_comb_i)
            offset += n_graph_i

        block_pair = np.concatenate(block_pair, axis=0)
        slice_batch_ptr = np.cumsum(slice_batch_ptr)
        iter_offset = index_black[np.concatenate(comb)].reshape(-1, 4)  # * 这样就不包含自己的边了
        return iter_offset, block_pair, slice_batch_ptr

    @classmethod
    def Block_Data_list_split_partition_list(cls, list_block: list[torch.Tensor], cross_slice, fn: Optional[Callable] = None):
        """
        把 list[Block_Data] 按照 cross_slice 中描述的方法进行切分，然后适配上 fn 函数进行转换，最后是变成符合 cross_slice 描述的 list[partition]
        """
        iter_offset, block_pair, slice_batch_ptr = cross_slice
        assert len(list_block) == len(slice_batch_ptr) - 1, "list_block 的长度必须和 slice_batch_ptr 的长度相同"

        offset_n = 0
        list_partition = []
        for i, block in enumerate(list_block):
            b_ptr_l = slice_batch_ptr[i]
            b_ptr_r = slice_batch_ptr[i + 1]
            iter_offset_i = iter_offset[b_ptr_l:b_ptr_r] - offset_n
            for iter_offset_ij in iter_offset_i:
                tmp_result = block[iter_offset_ij[0] : iter_offset_ij[1], iter_offset_ij[2] : iter_offset_ij[3]]
                if fn is not None:
                    tmp_result = fn(iter_offset_ij, tmp_result)
                list_partition.append(tmp_result)
            offset_n += len(block)
        return list_partition

    @classmethod
    def Block_Data_list_merge_from_partition_list(cls, list_partition, cross_slice, fn: Optional[Union[Callable, List[Callable]]] = None):
        """
        Args:
            list_matrix: 矩阵列表，可以是不同 Block_Data 切分成的交叉相关性矩阵，只需要在 slice_batch_ptr 中说明这些交叉矩阵分别属于哪一个 Block_Data 即可
            iter_offset: 矩阵列表对应的偏移量
            slice_batch_ptr: 说明哪些是一个 Block_Data 中的数据

        example:

            iter_offset = [[ 0  4  4  4]
                           [ 0  4  4  4]
                           [ 0  4  4  4]
                           [ 4  4  4  4]
                           [ 4  4  4  4]
                           [ 4  4  4  4]
                           [ 4  9  9 10]
                           [ 4  9 10 11]
                           [ 4  9 11 12]
                           [ 9 10 10 11]
                           [ 9 10 11 12]
                           [10 11 11 12]
                           [12 17 17 18]
                           [12 17 18 19]
                           [12 17 19 20]
                           [17 18 18 19]
                           [17 18 19 20]
                           [18 19 19 20]]

            slice_batch_ptr = [ 0  6 12 18]

        返回:
            一个列表，列表中的每一个元素是一个 Block_Data
        """
        iter_offset, block_pair, slice_batch_ptr = cross_slice
        num_graph = len(slice_batch_ptr) - 1
        list_Block_Data = []
        gen_partition = iter(list_partition)
        for k in range(num_graph):
            # 默认复原的是一个方阵，左上角的边缘索引是
            offset_merge_k = iter_offset[slice_batch_ptr[k] : slice_batch_ptr[k + 1]]
            i_0 = offset_merge_k[:, 0].min()
            j_0 = offset_merge_k[:, 1].min()
            i_1 = offset_merge_k[:, 2].max()
            j_1 = offset_merge_k[:, 3].max()
            p_0 = min(i_0, j_0)
            p_1 = max(i_1, j_1)
            dp = int(p_1 - p_0)  # type: ignore

            block_k = torch.zeros((dp, dp), requires_grad=True).to(list_partition[0])
            for pos in offset_merge_k:
                tmp_partition = next(gen_partition)
                ori_iijj = pos - p_0

                # * code copy ------------------------
                if fn is not None:
                    if isinstance(fn, list):
                        for fn_i in fn:
                            tmp_partition = fn_i(pos, tmp_partition)
                    else:
                        tmp_partition = fn(pos, tmp_partition)
                # * code copy ------------------------
                # * 可能会发生增广，所以后面依旧进行裁剪
                block_k[ori_iijj[0] : ori_iijj[1], ori_iijj[2] : ori_iijj[3]] = tmp_partition[0 : ori_iijj[1] - ori_iijj[0], 0 : ori_iijj[3] - ori_iijj[2]]

            list_Block_Data.append(block_k)
        return list_Block_Data

    @classmethod
    def Batch_Block_merged_partition_calc_to_blocklist(
        cls,
        matrix,
        cross_slice: Tuple[np.ndarray, np.ndarray, np.ndarray],
        fn: Union[Callable, List[Callable]],
    ):
        """
        传入一个 batch 化的稀疏大矩阵，然后分块计算，对每一个 piece 是一个 Block_Data，每一个 Block_Data 是多个相机构成的超图，
        根据，cross_slice的切分方式，将超图的邻接矩阵进行分块，用 fn 函数去计算每一个分块矩阵，然后将结果合并起来，未处理的分块以 0 补全
        最后将 batch 中的每一个 piece 合并成一个列表，列表长度即为 batch_size
        """

        list_partition = cls.Batch_Block_split_partition(matrix, cross_slice=cross_slice)
        list_block = cls.Block_Data_list_merge_from_partition_list(list_partition, cross_slice, fn=fn)

        return list_block

    @classmethod
    def Batch_Block_split_partition(cls, matrix, cross_slice: Tuple[np.ndarray, np.ndarray, np.ndarray], fn: Optional[Union[Callable, List[Callable]]] = None) -> list:
        """
        Args:
            matrix (torch.Tensor): 批次化的 Block，是多个 Block 组成的批次化大矩阵
            fn (Callable): 处理用的函数，遵从如下
                - ``lambda pos, adj: ... ``
                - fn 函数，有两个输入的参数，分别是 **for** 循环中的 ``pos`` 和 ``tmp_adj_ij``
                - fn 函数的返回值可以任意，最终作为list中的元素输出，但是要注意是其顺序是由 ``cross_slice`` 中的 ``iter_offset`` 来决定的

        fn 计算上三角交叉矩阵中的 基于行的topK的示例

        .. code-block::

            foo = lambda pox, m: get_topk_index(m.toarray(), K, axis=1) + pox[[0, 2], None]

        fn 计算上三角交叉矩阵中的 基于列的topK的示例

        .. code-block::

            foo = lambda pox, m: get_topk_index(m.toarray(), K, axis=0) + pox[[0, 2], None]

        """

        assert cross_slice is not None
        iter_offset, _, _ = cross_slice
        assert isinstance(matrix, torch.Tensor)

        if matrix.is_sparse:
            s_matrix, s_data = torch_sparse.from_torch_sparse(matrix.coalesce())
            sparse_matrix = torch_sparse.SparseTensor(row=s_matrix[0], col=s_matrix[1], value=s_data, sparse_sizes=matrix.shape)  # type: ignore

        list_result = []
        for pos in iter_offset:
            # * code copy ------------------------
            if matrix.is_sparse:
                tmp_result = sparse_matrix[pos[0] : pos[1], pos[2] : pos[3]].to_dense()  # type: ignore
            else:
                tmp_result = matrix[pos[0] : pos[1], pos[2] : pos[3]]
            if fn is not None:
                if isinstance(fn, list):
                    for fn_i in fn:
                        tmp_result = fn_i(pos, tmp_result)
                else:
                    tmp_result = fn(pos, tmp_result)
            # * code copy ------------------------
            list_result.append(tmp_result)

        return list_result


class Utils_Edge:
    """
    生成边的工具类
    """

    @staticmethod
    def edge_index_filter_by_set(edge_index: np.ndarray, block_set: tuple[set[int], set[int]]) -> np.ndarray:
        set_a = block_set[0]
        set_b = block_set[1]
        list_edge = []
        for i, j in edge_index.T:
            if i in set_a and j in set_b:
                list_edge.append((i, j))

        if len(list_edge) == 0:
            return np.zeros((2, 0), dtype=int)
        else:
            return np.array(list_edge).T

    @staticmethod
    def edge_index_filter_by_ptr(edge_index: np.ndarray, np_cumsum: np.ndarray, block_indexs: tuple[int, int]) -> np.ndarray:
        """
        return edge_index between [a, b), 这一函数的功能要求节点的索引号是聚类排序过的

        #todo 这个部分如果运行占用速度太多，可以改写成 cython

        example:
            >>> edge_index = np.array([[ 0,  2,  3,  3,  4,  4,  6, 10, 11, 12],
                                        [13,  9, 10, 14, 11, 15, 16, 14, 15, 20]], dtype=int32)
            >>> np_cumsum = np.array([ 0,  8, 13, 21]) # 3 个分块的 ptr 位置
            >>> edge_index_filter_by_ptr(edge_index, np_cumsum, (0, 1)) #* 获取图结构中 第 0 个分块和第 1 个分块的交叉边索引号

            [[ 2  3  4]
            [ 9 10 11]]
        """
        area_al, area_ar = np_cumsum[block_indexs[0]], np_cumsum[block_indexs[0] + 1]
        area_bl, area_br = np_cumsum[block_indexs[1]], np_cumsum[block_indexs[1] + 1]

        list_edge = []
        for i in range(edge_index.shape[1]):
            if (area_al <= edge_index[0, i] < area_ar) and (area_bl <= edge_index[1, i] < area_br):
                list_edge.append(i)
        if len(list_edge) == 0:
            return np.zeros((2, 0), dtype=int)
        else:
            return edge_index[:, np.array(list_edge)]

    @staticmethod
    def generate_allconnected_edge(shape: tuple[int, int], device=None):
        """
        生成边的列表
        :param batch_data:
        :return:
        """

        if device is None:
            device = torch.device("cpu")
        indexes_n = torch.arange(shape[0], device=device).expand(shape[1], shape[0]).T
        indexes_m = torch.arange(shape[1], device=device).expand(shape[0], shape[1])
        edge_index = torch.stack([indexes_n, indexes_m], dim=0)
        return edge_index.view(2, -1)

    @staticmethod
    def edgeindex_to_scipy_sparse(
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> sci_sparse.coo_matrix:
        """
        将边索引转化为 scipy sparse matrix

        .. warnings::
            :category: Deprecated
            这个只能设定一个节点，即只能生成 **正方形** 的稀疏矩阵


        Args:
            edge_index (Tensor):
            edge_attr (Tensor):
            num_nodes (int): 节点个数
        """
        pass
        return pyg_utils.to_scipy_sparse_matrix(edge_index, edge_attr, num_nodes)


def block_mask(len_block: tuple[list[int], list[int]]):
    """
    生成 二维分块矩阵的分块掩膜，返回的是一个稀疏矩阵

    Args:
        len_block (tuple[list[int], list[int]]): 第一个参数是每个维度分块的长度
        第二个参数是块索引号，说明哪些分块标记为 1，其余的标记为 0

    Returns:
    torch.Tensor:
    """
    pass
    raise NotImplementedError


from pretty_tools.datastruct.multi_index_dict import mdict


def merge_affinity(mdict_aff: mdict[tuple[int, int], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    构建一个全部的图连接结构, 但是没有想好放到哪个大类中进行管理

    输出合并后的矩阵，以及各处分块的位置
    ! 由于是mdict构成的，健的组成与顺序无关，因此aff将是一个无向的亲合度
    """
    assert mdict_aff.dim == 2
    np_len = np.zeros((mdict_aff.num_items,), int)
    list_aff: list[list[np.ndarray]] = [[None for i in range(mdict_aff.num_items)] for i in range(mdict_aff.num_items)]  # type: ignore
    for (i, j), v in mdict_aff.items():
        if np_len[i] == 0:
            np_len[i] = v.shape[0]
        if np_len[j] == 0:
            np_len[j] = v.shape[1]

        list_aff[i][j] = v
        list_aff[j][i] = v.T

        if list_aff[i][i] is None:
            list_aff[i][i] = np.zeros((v.shape[0], v.shape[0]))
        if list_aff[j][j] is None:
            list_aff[j][j] = np.zeros((v.shape[1], v.shape[1]))

    np_merge = np.block(list_aff)
    return np_merge, np_len
