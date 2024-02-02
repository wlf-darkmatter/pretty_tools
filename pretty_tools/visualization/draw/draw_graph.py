import io
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def visualize_graph(G, color):
    import networkx as nx

    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False, node_color=color, cmap="Set2")
    plt.show()


def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f"Epoch: {epoch}, Loss: {loss.item():.4f}", fontsize=16)
    plt.show()


class Visual_Graph:
    """
    可视化图结构的工具类
    #! 这个和跟踪类没有关系，不要相互耦合
    """

    def __init__(self, img: Optional[Image.Image] = None) -> None:
        import matplotlib as mpl
        import networkx as nx

        self.ratio_base = 10 / 1080

        self.ratio_nodesize = 0.5  # * 这个默认值是以开根的锚框面积来计算的， （1920,1080）图像，开根面积大概在 [80, 190] 之间
        self.ratio_arrowsize = 2
        self.width_edge = 3
        self.input_nodesizes: Union[int, np.ndarray] = 20
        self.cmap = mpl.colormaps["autumn"]  # todo 默认的颜色映射，后续应当设置成可通过方法来配置
        self._wh = (1920, 1080)  # * 默认值
        self.img = img

        self.G = nx.DiGraph()  #! 默认创建的是一个有向图 ，而不是一个 nx.Graph()
        self.pos: Optional[Union[np.ndarray, dict[Any, Tuple[float, float]], np.ndarray]] = None
        self.dist: Optional[np.ndarray] = None
        self.edge_color: Optional[Union[np.ndarray, int, float]] = None

    @property
    def wh(self):
        if self.img is not None:
            if isinstance(self.img, Image.Image):
                self._wh = (self.img.width, self.img.height)
            if isinstance(self.img, np.ndarray):
                self._wh = (self.img.shape[1], self.img.shape[0])
        return self._wh

    def set_node(self, node_list):
        self.G.add_nodes_from(node_list)

    def set_edge(self, edge_list):
        self.G.add_edges_from(edge_list)

    def set_dist(self, dist_list):
        self.dist = dist_list

    def set_pos(self, pos: "np.ndarray | Dict[Any, Tuple[float, float] | np.ndarray]"):
        # pos: (X,Y)
        if isinstance(pos, np.ndarray):
            assert pos.max() <= 2 and pos.min() >= -1, "调用 Visual_Graph()类进行可视化时，要注意 pos 位置坐标必须是长宽归一化的"
        elif isinstance(pos, dict):
            tmp_pos = np.array(list(pos.values()))
            assert tmp_pos.max() <= 2 and tmp_pos.min() >= -1, "调用 Visual_Graph()类进行可视化时，pos 位置坐标必须是长宽归一化的"
        self.pos = pos

    def set_node_size(self, node_sizes):
        self.input_nodesizes = node_sizes

    @property
    def num_nodes(self):
        return self.G.number_of_nodes()

    @property
    def num_edges(self):
        return self.G.number_of_edges()

    def set_edge_color(self, edge_color: "np.ndarray | int | float"):
        self.edge_color = edge_color
        if isinstance(self.edge_color, np.ndarray):
            if self.edge_color.ndim == 1:
                self.edge_color = self.cmap(edge_color)
            elif self.edge_color.shape[1] == 1:
                self.edge_color = self.cmap(edge_color)

            if self.edge_color.shape[1] == 3:
                self.edge_color = np.concatenate([self.edge_color, np.ones([self.edge_color.shape[0], 1])], axis=1)

    def draw_graph(self, img_camera: "None | Image.Image" = None, ax=None, new=True) -> Image.Image:
        import networkx as nx

        # * 如果 new 为 True，则新建一个画布
        if img_camera is None:
            img_camera = self.img
        else:
            self.img = img_camera
            #! 如果传入了 img_camera，则输出的图像中可视化的原图像的尺寸最好和传入的图像尺寸相仿
        refine_size = self.wh[1] * self.ratio_base

        if new:
            base_size = self.ratio_base * np.array(self.wh)
            fig = plt.figure(figsize=base_size, dpi=1 / self.ratio_base)

            fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            fig.gca().axis("off")

            fig.gca().margins(0, 0)
        if ax is None:
            ax = plt.gca()

        pos = deepcopy(self.pos)
        if img_camera is not None:
            # * 如果显示了原图，则pos需要使用原尺寸，而不是归一化的尺寸
            if isinstance(self.pos, np.ndarray):
                pos *= [img_camera.width, img_camera.height, img_camera.width, img_camera.height]  # * 这里的位置应当顺应图上的位置
            if isinstance(self.pos, dict):
                pos = {k: v * np.array([img_camera.width, img_camera.height]) for k, v in pos.items()}
        else:  #! 如果没有传入图片，依旧按照图像的格式进行显示，所以要进行倒转
            if isinstance(self.pos, np.ndarray):
                pos[:, 1] = 1 - pos[:, 1]
            elif isinstance(self.pos, dict):
                pos = {k: (v[0], 1 - v[1]) for k, v in pos.items()}

        nodes = nx.draw_networkx_nodes(
            self.G,
            pos,  #! pos 应当是一个能够被 node_list 的元素索引到的数组或者字典
            node_size=self.input_nodesizes * self.ratio_nodesize * refine_size,
            node_color="indigo",
            ax=ax,
        )

        # * 调节边的颜色，由长度决定
        if self.edge_color is None:
            if self.dist is not None:
                edge_colors = deepcopy(self.dist)
            else:
                edge_colors = np.arange(self.num_edges, dtype=np.float16)
            edge_colors -= edge_colors.min()
            edge_colors /= edge_colors.max()
            edge_colors = (edge_colors * 255).astype(int)
            self.edge_color = self.cmap(edge_colors)

        # * 上下两边都出现了 node_size ，上面的决定了节点的大小，下面的决定了连接线有多贴合节点
        # * 这里使得图像出现了白边
        edges = nx.draw_networkx_edges(
            self.G,
            pos,
            node_size=self.input_nodesizes * self.ratio_nodesize * refine_size,
            arrowstyle="-|>",
            arrowsize=self.ratio_arrowsize * refine_size,
            width=self.width_edge,
            edge_color=self.edge_color,
            ax=ax,
        )

        #! 图像的显示要放到最后，否则图像的尺寸在上一步可能会使得出现白边，不知道为什么
        if img_camera is not None:
            ax.imshow(img_camera)
        ax.set_axis_off()
        fig.set_size_inches(w=base_size[0], h=base_size[1])  # * 不加这个的话，显示某些比较大的图像的时候，会出现白边
        if new:
            fig.canvas.draw()  # 绘制图像
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format="jpg")
            image = Image.open(img_buf)
            return image
        else:
            return None  # type: ignore
