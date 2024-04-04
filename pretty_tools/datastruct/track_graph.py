from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib as jl
import numpy as np
from PIL import Image

try:
    from torch_geometric.data import Data
except:
    pass

from .general_ann import GeneralAnn


class TrackCameraGraph(GeneralAnn):
    """
    ! 改类要被弃用，ContionTrack==0.1.11.2

    单个相机检测到的目标组成的一个图结构

    初始化方法1

    .. code-block:: python

        from pretty_tools.datastruct import TrackCameraGraph
        tg = TrackCameraGraph.load(path)


    初始化方法2

    .. code-block:: python

        from pretty_tools.datastruct import TrackCameraGraph
        bbox1 = np.array([[0,0,100,100], [100,100,200,200]])
        general_ann = GeneralAnn(bbox1, str_format="ltrb")
        tg = TrackCameraGraph.from_general(general_ann)


    初始化方法3

    .. code-block:: python

        from pretty_tools.datastruct import TrackCameraGraph
        bbox1 = np.array([[0,0,100,100], [100,100,200,200]])
        edge_index = np.array([[0,1],[1,0]]) #! 可选
        tg = TrackCameraGraph(ann=bbox1, edge_index=edge_index, str_format="ltrb")


    -------

    节点特征

    :code:`tg.x`


    可视化
    :code:`tg.visualize()`

    """

    @classmethod
    def from_general(
        cls,
        from_general: GeneralAnn,
        node_x: Optional[np.ndarray] = None,
        edge_index: Optional[np.ndarray] = None,
        edge_attr: Optional[np.ndarray] = None,
    ) -> TrackCameraGraph:
        # todo 这个功能还没写测试函数
        return cls(
            from_general=from_general,
            node_x=node_x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

    def __init__(
        self,
        ann: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
        ori_ann: Optional[GeneralAnn] = None,
        str_format: Optional[str] = None,
        ori_WH: Optional[Tuple[int, int]] = None,
        ori_img: Optional[Image.Image] = None,
        from_general: Optional[GeneralAnn] = None,
        index_camera: int = -1,
        node_x: Optional[np.ndarray] = None,
        edge_index: Optional[np.ndarray] = None,
        edge_attr: Optional[np.ndarray] = None,
    ) -> None:
        """
        初始化的时候需要创建所有节点，但是边可以省略

        可以有各种各样的初始化方法，全部都在测试方法中进行测试

        edge_index 应当是索引对，注意，索引并非节点的id，而是节点在node_x中的索引位置
        """

        self.index_camera = index_camera
        if from_general is not None:
            #! 内部会要求除了 from_general 其他都是None
            super().__init__(
                from_general,
                None,
                str_format=str_format,
                ori_WH=ori_WH,
                ori_img=ori_img,
                _move_copy=True,
            )
            if ori_WH is not None and not self.flag_norm:
                self.set_ori_WH(ori_WH, renorm=True)
        else:
            super().__init__(ann, ori_ann, str_format=str_format, ori_WH=ori_WH, ori_img=ori_img)

        self.graph = Data()

        self.x = None
        self.edge_index = None

        if node_x is not None:
            self.set_node_x(node_x)
        if edge_index is not None:
            self.set_edge_index(edge_index)
        if edge_attr is not None:
            self.set_edge_attr(edge_attr)

        # * 获取 锚框的中心点作为节点位置（仅用于可视化）
        self.set_node_pos(self.xywh[:, :2])

    @property
    def node_x(self):
        return self.x

    @property
    def num_nodes(self) -> int:
        if self.x is not None:
            return len(self.x)
        else:
            return 0

    def set_node_x(self, node_x):
        import torch

        """
        传入的是特征矩阵，每一行是一个节点的特征
        """
        if node_x is None:
            raise ValueError("set_node_x(node_x)中， node_x 不能为 None")
        assert node_x.shape[0] == self.num_boxes
        if isinstance(node_x, np.ndarray):
            node_x = torch.as_tensor(node_x)
        self.graph["x"] = node_x
        self.x = node_x
        pass

    def set_node_pos(self, pos: np.ndarray):
        """
        传入的是节点的位置矩阵，每一行是一个节点的位置

        由于初始化的时候必须用到锚框，所以一定存在节点的坐标位置信息，以此为依据计算每个边的距离，自动分配默认的颜色
        """
        #! 由于在跟踪类中，一定是通过目标ID号进行索引的，所以这里 可视化的时候 一定要按照节点的名称，使用字典进行索引
        if pos is None:
            UserWarning("警告，使用了一个None手动构建节点位置")
            return

        if isinstance(pos, np.ndarray):
            import torch

            self.graph["pos"] = torch.as_tensor(pos)
        else:
            self.graph["pos"] = pos
        pass

    @property
    def pos(self):
        return self.graph["pos"]

    def set_edge_index(self, edge_index):
        if edge_index is None:
            UserWarning("警告，使用了一个None手动构建边关系")
            return
        if isinstance(edge_index, np.ndarray):
            import torch

            edge_index = torch.as_tensor(edge_index)

        if edge_index.shape[0] != 2:
            if edge_index.shape[1] == 2:
                UserWarning("edge_index shape warning, should be (2, num_edges), but get (num_edges, 2)， now transpose it")
                edge_index = edge_index.T
            else:
                raise ValueError("edge_index shape error, should be (2, num_edges)")
        self.edge_index = edge_index
        self.graph["edge_index"] = edge_index
        pass

    def set_edge_attr(self, edge_attr):
        if edge_attr is None:
            UserWarning("警告，使用了一个None手动构建边特征值")
            return
        # todo 缺少数据类型校验部分
        self.graph["edge_attr"] = edge_attr

    def validate(self, *args, **kwargs):
        return self.graph.validate(*args, **kwargs)

    # * -----------------  一些属性  -----------------
    def __repr__(self) -> str:
        info = f"{self.__class__.__name__}(num_nodes={len(self.ann)}, "
        if self.x is not None:
            info += f"dim_feature={self.x.shape[1]}, "
        if self.edge_index is not None:
            info += f"num_edges={self.edge_index.shape[1]}, "
        try:
            info += f"ori_wh={self.__ori_WH}, "
        except:
            pass
        return f"{info} \ndata=\n{self.ann}, )"

    # * ----------------- IO 功能 -----------------
    @staticmethod
    def save(ann_target: TrackCameraGraph, path_save: Union[str, Path]):
        path_save = Path(path_save)
        assert path_save.suffix == ".pkl", "必须使用 .pkl 结尾进行保存"  # * 必须使用joblib进行读写
        jl.dump(ann_target, path_save)

    @staticmethod
    def load(path_load: Union[str, Path]):
        """

        Parameters
        ----------
        path_load: str, pathlib.Path
        """
        ann_target = jl.load(path_load)
        assert isinstance(ann_target, TrackCameraGraph)
        return ann_target

    # * ----------------- 可视化功能 -----------------
    def visualize(
        self,
        img_camera: Optional[Image.Image] = None,
        node_sizes: Optional[np.ndarray] = None,
        edge_color: Optional[Union[int, np.ndarray, tuple, Sequence[tuple]]] = None,
        ax=None,
        new=True,
    ) -> Image.Image:
        """
        绘制图像，需要自动补全一些绘图信息

        Args:
            img_camera (Image.Image, optional): Image.Imgea 原图，可选
            node_sizes (np.ndarray, optional): 点的大小，可选
            new (bool, optional):  用于控制是否被上一层绘图再次封装，为True时表示这是一个全新的绘图，否则表示这是一个子图
        """
        from matplotlib import pyplot as plt
        from pretty_tools.visualization import Visual_Graph

        visual_graph = Visual_Graph()
        if (self.ids == self.ids[0]).all():
            node_list = np.array(np.arange(self.num_boxes))
        else:
            node_list = np.array(self.ids)
        visual_graph.set_node(node_list)

        pos = self.graph["pos"]
        if isinstance(pos, np.ndarray):
            pos = dict(zip(node_list, pos))
        else:
            pos = dict(zip(node_list, pos.numpy()))

        visual_graph.set_pos(pos)

        #! 这里给到 networkx里面的边，不再是索引号，而应该是名称列表，所以要根据名称变换一下
        assert self.edge_index is not None
        np_edge_index = self.edge_index.T.numpy()
        edge_list = np.array(node_list)[np_edge_index]
        visual_graph.set_edge(edge_list)  # * 绘图的时候，边的索引是 (num_edges, 2) 的形式

        if node_sizes is None:  # * 默认是用的节点大小计算方案
            node_sizes = np.sqrt(self.ori_xywh[:, 2] * self.ori_xywh[:, 3])
            visual_graph.set_node_size(node_sizes)
        if edge_color is not None:
            visual_graph.set_edge_color(edge_color)  # type: ignore
        else:
            # * 没有给定的颜色，自动使用对应锚框的距离进行颜色的分配
            delta_xy = self.ori_xywh[np_edge_index, :2]
            delta_xy = delta_xy[:, 0, :] - delta_xy[:, 1, :]
            dist = np.sqrt(np.sum(delta_xy**2, axis=1))
            dist -= dist.min()
            dist /= dist.max()
            dist = (dist * 255).astype(int)
            visual_graph.set_edge_color(dist)

        if img_camera is None:
            img_camera = self.ori_img
        img_graph = visual_graph.draw_graph(img_camera=img_camera, ax=ax, new=new)
        return img_graph

    @staticmethod
    def draw_match(
        list_graph_A: Union[Sequence[TrackCameraGraph], TrackCameraGraph],
        list_graph_B: Union[Sequence[TrackCameraGraph], TrackCameraGraph],
        list_match: Union[Sequence[np.ndarray], np.ndarray],
        list_gt_match: Optional[Union[Sequence[np.ndarray], Sequence[Tuple[int, int]], np.ndarray]] = None,
    ):
        """
        给入两组待对比的图实例，同时给出已经匹配好的索引，进行绘制
        list_match: (num_match, 2) 的形式 匹配列表
        list_gt_match 真值，可选
        """
        from matplotlib import pyplot as plt
        from matplotlib.patches import ConnectionPatch

        ratio_base = 10 / 1080
        if type(list_graph_A) is not list:
            assert type(list_graph_A) is TrackCameraGraph, f"Since list_graph_A type as not list, it should be TrackCameraGraph, but get {type(list_graph_A)}"
            assert type(list_graph_B) is TrackCameraGraph, f"Since list_graph_B type as not list, it should be TrackCameraGraph, but get {type(list_graph_B)}"
            assert type(list_match) is np.ndarray, f"Since list_match type as not list, it should be np.ndarray, but get {type(list_match)}"
            list_graph_A = [list_graph_A]
            list_graph_B = [list_graph_B]
            list_match = [list_match]
        assert type(list_graph_A) is list, f"list_graph_A type error, should be list, but get {type(list_graph_A)}"
        assert type(list_graph_B) is list, f"list_graph_B type error, should be list, but get {type(list_graph_B)}"
        assert type(list_match) is list, f"list_match type error, should be list, but get {type(list_match)}"
        num_show = len(list_graph_A)
        fig, axs = plt.subplots(
            num_show,
            2,
            figsize=(ratio_base * 2 * 1920, ratio_base * num_show * 1080),
            dpi=108,
        )
        fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.05, wspace=0.05)

        iter_ax = iter(axs)
        for graph_A, graph_B, match in zip(list_graph_A, list_graph_B, list_match):
            # * 放弃封装式绘图，直接将子图绘制好然后作为一个新的位图进行绘制
            ax1 = next(iter_ax)
            img_A = graph_A.visualize(new=True)
            ax1.imshow(img_A)
            ax2 = next(iter_ax)
            img_B = graph_B.visualize(new=True)
            ax2.imshow(img_B)
            pass
            # * 画连接线
            new_pos_A = deepcopy(graph_A.pos.numpy())
            new_pos_B = deepcopy(graph_B.pos.numpy())
            # * 上下倒转一下，否则绘图时候会出错
            new_pos_A[:, 1] = 1 - new_pos_A[:, 1]
            new_pos_B[:, 1] = 1 - new_pos_B[:, 1]
            if list_gt_match is None:
                for pair in match:
                    con = ConnectionPatch(
                        xyA=new_pos_A[pair[0], :],
                        xyB=new_pos_B[pair[1], :],
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax2,
                        color="green",
                    )
                    ax2.add_artist(con)
            else:
                list_gt_match = [tuple(i) if type(i) is not tuple else i for i in list_gt_match]
                for pair in match:
                    con = ConnectionPatch(
                        xyA=new_pos_A[pair[0], :],
                        xyB=new_pos_B[pair[1], :],
                        coordsA="axes fraction",
                        coordsB="axes fraction",
                        axesA=ax1,
                        axesB=ax2,
                        color="green" if tuple(pair) in list_gt_match else "red",
                    )
                    ax2.add_artist(con)

        return fig
