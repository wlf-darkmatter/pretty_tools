"""
.. note::

    torch 隐式依赖，只有通过显示调用才会强制使用torch

.. note::

    后续这里的绘制工具应当进行改进，不应该再使用 :class:`mdict` 了，mdict适合进行数据的写入，
    但是在进行处理的时候没法批量调用，之后写的工具函数应当规避掉这一点，不再使用 **mdict**

.. important::

    这个模块规定了全局的字体格式为 ``Times New Roman`` 字体


.. important::

    1. 绘图相关的数据建议全部采用 :class:`numpy` 或者 :class:`scipy.sparse.spmatrix` 类型
    2. 请不要使用 :class:`torch` 进行开发，因为有的项目并不安装 torch，当涉及到这个 draw 的 功能的时候，岂不是要强制安装一个 torch？
    建议通过判端 ``type(data) == "<class 'torch.Tensor'>"`` 来判端是否是 :class:`torch` 类型的数据，并通过调用 :func:`numpy()` 将其转换成 numpy 类型。

"""

import copy
import functools
import itertools
import math
import sys
from copy import deepcopy
from functools import partial
from typing import Any, Unpack, List, Literal, Optional, Sequence, Tuple, TypedDict, Union, Unpack

import cv2
import matplotlib
import matplotlib.figure
import matplotlib.transforms as mtransforms
import numpy as np
import seaborn as sns
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import ConnectionPatch
from matplotlib.pyplot import Figure
from pandas import DataFrame
from PIL import Image, ImageDraw, ImageFont

# * 处理部分需要使用到torch的库
try:
    import torch
except:
    pass
try:
    from pretty_tools.datastruct import graph_enhance, np_enhance
except:
    pass
from scipy import sparse

from pretty_tools.datastruct.X_bbox import dict_convert_fn
from pretty_tools.datastruct.cython_bbox import cy_bbox_overlaps_iou
from pretty_tools.datastruct.multi_index_dict import mdict
from pretty_tools.datastruct.np_enhance import convert_to_numpy
from pretty_tools.datastruct.numpy_bbox import bbox_no_overlaps_area
from pretty_tools.resources import path_font_arial, path_font_time_new_roman
from pretty_tools.solver.match_utils import match_result_check

from . import matplotlib_misc

matplotlib.use("Agg")

#! 按照规定，字体应当使用 新罗马字体
font = {"family": "serif", "serif": "Times New Roman", "weight": "normal", "size": 10}
plt.rc("font", **font)
font_time_new_roman = matplotlib.font_manager.FontProperties(fname=str(path_font_time_new_roman))


class Visiual_Tools:
    @staticmethod
    def convert_img(img):
        if type(img) == np.ndarray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        assert type(img) == Image.Image
        return img

    @staticmethod
    def get_wh(image: Union[np.ndarray, Image.Image]) -> Tuple[int, int]:
        if isinstance(image, np.ndarray):
            return image.shape[1::-1]  # type: ignore
        elif isinstance(image, Image.Image):
            return image.size

    @staticmethod
    def fig_to_image(fig):
        """
        将matplotlib的figure转化为PIL的Image
        """
        import io

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        image = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)

        return image

    @classmethod
    def _plot_block_line(cls, the_ax, shape, aux_linx_x=None, aux_linx_y=None, Oij=None):
        """
        调节多块网格的显示

        shape[0] 是行数，shape[1] 是列数

        ? 20240410 修改, 删去了 np_cumsum, 改而使用 aux_linx_x, aux_linx_y
        """

        if Oij is None:
            Oij = (0, 0)

        the_ax.set_xlim([Oij[1] - 0.5, Oij[1] + shape[1] - 0.5])
        the_ax.set_ylim([Oij[0] - 0.5, Oij[0] + shape[0] - 0.5])
        the_ax.set_xticks(np.arange(Oij[1], Oij[1] + shape[1]))  # 绘制网格，添加额外的边界线
        the_ax.set_yticks(np.arange(Oij[0], Oij[0] + shape[0]))  # 绘制网格，添加额外的边界线
        the_ax.grid(axis="both", which="both", linewidth=0.5, zorder=-10)  # 绘制网格，设定网格线的宽度
        the_ax.margins(0.05)  # * 5% 的空白

        if aux_linx_x is not None:
            if aux_linx_x[-1] == shape[1]:
                aux_linx_x = aux_linx_x[:-1]
            for pox_x in aux_linx_x:
                the_ax.axvline(x=pox_x - 0.5, dashes=[4, 4], zorder=-5)  # * 绘制垂直线

        if aux_linx_y is not None:
            if aux_linx_y[-1] == shape[1]:
                aux_linx_y = aux_linx_y[:-1]
            for pox_y in aux_linx_y:
                the_ax.axhline(y=pox_y - 0.5, dashes=[4, 4], zorder=-5)  # * 绘制水平线

        the_ax.axis("equal")

    @staticmethod
    def cut_bbox(
        image: Union[np.ndarray, Image.Image, Any],
        bbox: "np.ndarray",
        str_format="ltrb",
    ):
        """
        image 待切分的图像
        bbox 必须是整型，必须是像素值，不能是归一化的值
        """

        if type(bbox) not in [np.ndarray]:
            UserWarning(f"bbox 参数的类型为 {type(bbox)}, 并未测试过")
        bboxes: np.ndarray = dict_convert_fn[str_format]["ltrb"](bbox).astype(int)
        list_cut = []
        if isinstance(image, Image.Image):
            for bbox in bboxes:
                list_cut.append(image.crop(bbox))  # type: ignore
        elif isinstance(image, np.ndarray):
            for bbox in bboxes:
                list_cut.append(image[bbox[1] : bbox[3], bbox[0] : bbox[2]])
        else:  # * 这里应当作为 torch.Tensor 处理
            for bbox in bboxes:
                bbox = bbox.astype(int)
                list_cut.append(image[:, bbox[1] : bbox[3], bbox[0] : bbox[2]])

        return list_cut


class Dict_Kwargs_GridSpec(TypedDict):
    figure: matplotlib.figure.Figure
    left: float
    bottom: float
    right: float
    top: float
    wspace: float
    hspace: float
    width_ratios: Sequence[float]
    height_ratios: Sequence[float]


class Dict_Kwargs_scatter(TypedDict):
    """
    Dict_Kwargs_scatter 棋盘格绘图参数


    """

    aux_linx_X: np.ndarray | Sequence[int]  # *竖直的辅助线的值. 输入的应当是 x 轴的坐标
    aux_linx_y: np.ndarray | Sequence[int]  # *水平的辅助线的值，输入的应当是 y 轴的坐标
    shape: tuple[int, int]  # *如果输入的是 稀疏矩阵 或 列表，这个 shape 则建议输入进来. 默认使用 ``adj`` 的尺寸
    scatter_sizes: tuple[int, int]  # *散点图的大小范围 (5, 80)
    label_x: str  # *x轴的标签，默认为"$j$"
    label_y: str  # *y轴的标签，默认为"$i$"
    Oij: tuple[int, int]  # *左上角偏移原点的位置，默认为 Oij=(x=0, y=0), x 表示左右，y 表示上下
    dpi: int  # *dpi，默认为200
    size_norm: tuple[int, int]
    hide_triu: bool  # *是否遮挡上面的三角，默认为 False
    hide_tril: bool  # *是否遮挡下面的三角，默认为 False
    hide_diag: bool  # *是否遮挡 对角线分块矩阵，默认为 False
    mid_label_x: Sequence[str]  # * 标记在每一个 aux_linx_y 上的标签
    mid_label_y: Sequence[str]  # * 标记在每一个 aux_linx_y 上的标签

    cm_aff: str  # *颜色条的名称，默认为 "inferno_r"

    fig_clear: bool  # ?查阅 draw_edge_index 函数参数说明


class Pretty_Draw:
    outline = 2
    dpi = 200

    @staticmethod
    def draw_affinity(
        mdict_similarity: mdict,  # 记录两两之间的亲合度矩阵
        dict_img: Union[dict[Any, Image.Image], dict[Any, np.ndarray]] = None,  # type: ignore #todo 这里的类型声明有问题
        dict_ltrb: dict[Any, np.ndarray] = None,  # type: ignore
        dict_list_cut: dict[Any, list[Union[Image.Image, np.ndarray]]] = None,  # type: ignore
        mdict_pair: mdict = None,  # type: ignore
        mdict_pair_gt: mdict = None,  # type: ignore
        reid_shape: tuple[int, int] = (128, 256),  # * (w,h)
        dpi=200,
    ):
        """
        绘制亲合度矩阵

        dict_ltrb 必须是未归一化的，以后做统一规定，做图像的可视化时，所有锚框都必须是未归一化的

        mdict_pair_gt 是GT值

        显示的结果中，加粗且带有边框的是真值
        斜体的是推理值

        todo 之后，判断是否匹配正确，如果推理的匹配和真值相匹配，则在边框的右下角打上一个绿色的✅

        Example
        -------

        .. code-block:: python

            # 可视化两张图的亲合度矩阵
            from pretty_tools.datastruct import mdict
            mdict_similarity = mdict()
            mdict_similarity[0, 1] = np.random.rand(10, 10) # 生成一个 10*10 的亲合度矩阵
            fig = Pretty_Draw.draw_affinity(mdict_similarity)


        """
        from matplotlib.offsetbox import AnnotationBbox, DrawingArea, OffsetImage, TextArea

        z_t = 0.8
        _d_z_t = (1 - z_t) / 2
        assert (dict_img is not None) == (dict_ltrb is not None), "dict_img 和 dict_ltrb 必须同时不为None 或者同时为None"

        ratio_hw = reid_shape[1] / reid_shape[0]
        assert mdict_similarity.dim == 2, "mdict必须是二维的"

        max_m = sum([i.shape[1] for i in mdict_similarity.values()])
        max_n = max([i.shape[0] for i in mdict_similarity.values()])

        if dict_img is not None:
            # * 如果没有传入切分后的图像，则进行切分（需要传入ltrb锚框信息）
            dict_list_cut = {}
            for camera_name, img in dict_img.items():
                dict_list_cut[camera_name] = Visiual_Tools.cut_bbox(img, dict_ltrb[camera_name])
        if dict_list_cut is not None:
            # * 如果存在切分后的图像，按照要求进行resize
            for list_cut in dict_list_cut.values():
                for i, cut in enumerate(list_cut):
                    if isinstance(cut, Image.Image):
                        list_cut[i] = cut.resize(reid_shape)
                    else:
                        list_cut[i] = cv2.resize(cut, reid_shape)
        fig, axs = plt.subplots(nrows=1, ncols=len(mdict_similarity), figsize=(max_m, max_n), dpi=dpi)
        fig.subplots_adjust(top=1 - _d_z_t - 0.05, bottom=_d_z_t, right=1, left=_d_z_t)  # * top 多减的 0.05是为了放图像
        # todo 针对 len(mdict_similarity) == 1 的情况，设计一个test
        if len(mdict_similarity) == 1:
            axs = [axs]

        for i, ((camera1, camera2), matrix) in enumerate(mdict_similarity.items()):
            the_ax = axs[i]
            annot_kws = {"fontsize": 18}
            # annot_kws = {}
            draw_heatmap = partial(
                sns.heatmap,
                annot=True,
                cmap="coolwarm",
                vmin=0,
                vmax=1,
                ax=the_ax,
                annot_kws=annot_kws,
            )
            if i != len(mdict_similarity):
                the_ax = draw_heatmap(matrix, cbar=False)  # * 使用统一的颜色条
            else:
                the_ax = draw_heatmap(matrix, cbar_ax=[axs[i]])  # * 使用统一的颜色条
            # ----------------------------------
            zw = dpi / reid_shape[0] / matrix.shape[1]  ## matrix.shape[1] # 是相似度矩阵水平方向的尺寸
            zh = dpi / reid_shape[1] / matrix.shape[0]  ## matrix.shape[0] # 是相似度矩阵垂直方向的尺寸
            # * 获取当前网格的大小
            if dict_list_cut is not None:
                for t, image in enumerate(dict_list_cut[camera2]):  # * 绘制每列对应的图像，（上方）
                    imagebox_w = OffsetImage(image, zoom=zw * z_t / 0.7)  # * 0.7可能是基准比例
                    imagebox_w.image.axes = the_ax  # type: ignore
                    the_ax.add_artist(
                        AnnotationBbox(
                            imagebox_w,
                            (t + 0.5, 0),
                            xybox=(t + 0.5, -0.3 * zw / zh),
                            frameon=False,
                        )
                    )
                for t, image in enumerate(dict_list_cut[camera1]):  # * 绘制每行对应的图像，（左侧）
                    imagebox_h = OffsetImage(image, zoom=zh * ratio_hw * z_t / 0.7)
                    imagebox_h.image.axes = the_ax  # type: ignore
                    the_ax.add_artist(
                        AnnotationBbox(
                            imagebox_h,
                            (0, t + 0.5),
                            xybox=(-1 * zh / zw, t + 0.5),
                            frameon=False,
                        )
                    )
            if mdict_pair_gt is not None:  # * 高亮匹配的框
                for pair in np.array(mdict_pair_gt.loc[camera1, camera2]):
                    text = the_ax.texts[int(pair[0] * matrix.shape[1] + pair[1])]
                    text.set_size(20)
                    text.set_style("italic")
                    text.set_bbox(dict(pad=0, ec="k", fc="none"))
            if mdict_pair is not None:  # * 高亮匹配的框
                for pair in np.array(mdict_pair.loc[camera1, camera2]):
                    text = the_ax.texts[int(pair[0] * matrix.shape[1] + pair[1])]
                    text.set_size(20)
                    text.set_weight("bold")
                    text.set_bbox(dict(pad=0, ec="k", fc="none"))
                    # text.set_verticalalignment("baseline")
            the_ax.set_xlabel(camera2, labelpad=0)
            the_ax.set_ylabel(camera1, labelpad=80 * zh / zw)
            the_ax.yaxis.tick_right()  # * y轴标签放在右边

        fig.colorbar(axs[-1].collections[0], ax=axs, location="right")  # type: ignore # 设置整个画布的颜色条
        return fig
        # fig.savefig("tmp/affinity.jpg")
        # plt.savefig("tmp/affinity.jpg")

    @staticmethod
    def draw_bboxes(
        img,
        bboxes_ltrb: np.ndarray,
        ids: Optional[list[int]] = None,
        colors: Optional[Union[int, tuple[int, int, int], list[tuple]]] = None,  # RGB,
        infos: Optional[list] = None,
        outline=2,
        size_font=24,
        mask: float = 0,
    ) -> Image.Image:
        """
        传入一张图，以及一张图的标注信息，进行绘制

        Args:

            bboxes : LTRB 格式
            ids 可选
            colors : 如果是使用统一的颜色，则需要输入的 colors 应当为 tuple
        """
        #! 仅支持这一种标注格式，不再做其他的适配（太麻烦，而且效率不高）
        assert bboxes_ltrb.ndim == 2, "格式必须是二维数组，长度为(n, 4)"
        assert bboxes_ltrb.shape[1] == 4
        if colors is None:
            colors = (255, 255, 255)
        if isinstance(colors, list):
            assert len(colors) == len(bboxes_ltrb), "如果是使用统一的颜色，则需要输入的 colors 应当为 tuple"

        bboxes_ltrb = copy.deepcopy(bboxes_ltrb)
        #! 一定要是np.float32或者直接用int
        if bboxes_ltrb.dtype == np.float64 or bboxes_ltrb.dtype == np.float128:
            bboxes_ltrb = bboxes_ltrb.astype(np.float32)
        bboxes_ltrb = bboxes_ltrb.astype(np.float32)

        if infos is not None:
            assert len(bboxes_ltrb) == len(infos)
        if ids is not None:
            assert len(bboxes_ltrb) == len(ids)

        if isinstance(img, np.ndarray):
            image = Image.fromarray(img)
        elif isinstance(img, Image.Image):
            image = copy.deepcopy(img)
            pass
        else:  # * 视作作为 torch.Tensor
            image = Image.fromarray(np.transpose(img.detach().cpu().numpy(), (1, 2, 0)))

        if bboxes_ltrb.ndim == 1:
            bboxes_ltrb = bboxes_ltrb[np.newaxis, :]

        if type(colors) == tuple:
            colors = len(bboxes_ltrb) * [colors]

        # * 绘制具有透明度的图片

        if mask != 0:
            mask = int(255 * mask)
            _image = image.convert("RGBA")
            image = Image.new("RGBA", img.size, (0, 0, 0, 0))  # type: ignore

            colors = [(*color, 255) for color in colors]  # type:ignore
            #! 如果有透明度，则自上而下地绘制，下面的覆盖上面的
            sort_index = np.argsort(bboxes_ltrb[:, 3])
            colors = [colors[i] for i in sort_index]
            infos = [infos[i] for i in sort_index] if infos is not None else infos
            ids = [ids[i] for i in sort_index] if ids is not None else ids

            bboxes_ltrb = bboxes_ltrb[sort_index]
            # * 绘制mask
            draw = ImageDraw.Draw(image, mode="RGBA")
            for box, color in zip(bboxes_ltrb, colors):
                draw.rectangle(box, width=outline, outline=color, fill=color[:3] + (mask,))  # type: ignore
            image = Image.alpha_composite(_image, image)
            image = image.convert("RGB")

        draw = ImageDraw.Draw(image, mode="RGB")
        # * 绘制边框 不能和 mask 一起绘制，会出现边框被mask覆盖的问题
        for box, color in zip(bboxes_ltrb, colors):  # type: ignore
            draw.rectangle(box, width=outline, outline=color)

        if infos is not None:
            font = ImageFont.truetype(str(path_font_arial), size_font)
            for box, info in zip(bboxes_ltrb, infos):
                draw.text(box[2:4], info, font=font)  # * info 绘制在 锚框的 右下角
        if ids is not None:
            font = ImageFont.truetype(str(path_font_arial), size_font)
            for box, _id in zip(bboxes_ltrb, ids):
                draw.text(box[0:2], str(_id), font=font)  # * info 绘制在 锚框的 左上角
        return image

    @staticmethod
    def draw_bboxes_matplotlib(
        img: Union[np.ndarray, Image.Image],
        np_ltrb: np.ndarray,
        ids: Optional[np.ndarray] = None,  # type: ignore
        str_format: str = "ltrb",  # type: ignore
    ):
        """
        img 输入的图像，可以是cv2格式的，也可以是PIL格式的
        默认输入的是 LTRB，但是通过 str_format 说明格式，会自动进行转换

        返回的是一个fig对象
        """
        import matplotlib.patches as patches

        assert isinstance(np_ltrb, np.ndarray), "必须是ndarray"
        assert np_ltrb.ndim == 2
        assert np_ltrb.shape[1] == 4
        np_ltwh = dict_convert_fn[str_format]["ltwh"](np_ltrb)

        w, h = Visiual_Tools.get_wh(img)
        fig = plt.figure(figsize=(w / Pretty_Draw.dpi, h / Pretty_Draw.dpi), dpi=Pretty_Draw.dpi)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        # fig.figimage(img)  # type: ignore
        ax = fig.gca()
        ax.imshow(img)  # type: ignore
        for ltwh in np_ltwh:
            # xywh = xywh / np.array([w, h, w, h])
            rectangle = patches.Rectangle(ltwh[:2], ltwh[2], ltwh[3], linewidth=1, edgecolor="r", facecolor="none")
            ax.add_patch(rectangle)
            pass
        # fig.savefig("tmp/tmp.jpg")
        return fig

    @staticmethod
    def draw_combine_affinity_mdict(
        mdict_similarity: mdict,
        mdict_pair: mdict = None,  # type: ignore
        mdict_pair_gt: mdict = None,  # type: ignore
        dict_edge: dict = None,  # type: ignore #* 用于可视化的图连接
        *,
        mdict_pair_dev: mdict = None,  # type: ignore #* 开发中的可视化
    ):
        """
        和 draw_affinity() 有点类似，绘制的是一个拓扑亲合度矩阵，但是为了方便不再显示目标锚框的截图

        Args:
            mdict_similarity : "mdict必须是二维的", 之后会改成可以使用一个大矩阵，现阶段不进行处理

        .. code-block:: python

            # 可视化三张图的亲合度矩阵
            from pretty_tools.datastruct import mdict
            mdict_similarity = mdict(2)
            mdict_similarity[0, 1] = np.random.rand(10, 10) # 生成一个 10*10 的亲合度矩阵
            mdict_similarity[0, 2] = np.random.rand(10, 10) # 生成一个 10*10 的亲合度矩阵
            mdict_similarity[1, 2] = np.random.rand(10, 10) # 生成一个 10*10 的亲合度矩阵
            fig = Pretty_Draw.draw_combine_affinity(mdict_similarity)


        .. note::

            这个调用可能有点困难，因为不一定就是使用的 mdict，使用前还要转换成 mdict。
            todo 之后根据这一点进行更改


        """
        import math

        import matplotlib
        import seaborn as sns
        from pandas import DataFrame

        assert mdict_similarity.dim == 2, "mdict必须是二维的"
        # * 必须是两两的亲合度矩阵，校验矩阵个数
        n = math.ceil(math.sqrt(2 * len(mdict_similarity)))
        assert math.comb(n, 2) == len(mdict_similarity), "mdict_similarity 的长度不符合要求"
        n_fig = 1
        if mdict_pair is not None:
            n_fig += 1
        if mdict_pair_dev is not None:
            n_fig += 1
        #! 合并矩阵
        merge_aff, np_len = graph_enhance.merge_affinity(mdict_similarity)

        # * 处理不太好处理的 mdict_similarity
        # ----------------------------------------

        sum_node = int(sum(np_len))
        np_cumsum = np.cumsum(np.insert(np_len, 0, 0))

        curt_fig = 0
        max_shape_size = sum_node**2
        fig, axes = plt.subplots(1, n_fig, figsize=(sum_node / 3 * n_fig, sum_node / 3), dpi=Pretty_Draw.dpi)
        df = DataFrame(np_enhance.index_value_2d(merge_aff), columns=["x", "y", "value"])
        #! 这里绘制 相似度部分的节点图
        ax = axes[0] if n_fig != 1 else axes
        cm_aff = matplotlib.colormaps["plasma_r"]  # type: ignore  # tab20b_r # tab20c_r
        scatter_sizes = (5, max_shape_size * 0.5)
        ax = sns.scatterplot(
            data=df,
            x="x",
            y="y",
            size="value",
            hue="value",
            ax=ax,
            palette=cm_aff,
            sizes=scatter_sizes,
            zorder=2,
        )
        sns.move_legend(ax, "upper left", labelspacing=sum_node / 20, ncol=1, frameon=True, bbox_to_anchor=(1, 1), borderaxespad=0)

        Visiual_Tools._plot_block_line(ax, merge_aff.shape, np_cumsum)
        ax.xaxis.set_ticks_position("top")  # * 将x轴的位置设置在顶部
        ax.tick_params(axis="both")
        ax.invert_yaxis()  # * y轴反向
        # * 绘制度矩阵
        if dict_edge is not None:
            combine_list_edge = [v + np_cumsum[k] for k, v in dict_edge.items()]
            np_edge = np.concatenate(combine_list_edge, axis=1)
            del combine_list_edge
            # * 这里xy要互换一下，散点图的横轴是x轴，但是边索引关系的第0行是纵轴
            ax.scatter(np_edge[1], np_edge[0], marker=r"$e$", c="green", s=80, zorder=3)

        dict_gt: dict[int, int] = {}  # *  dict{global_index: global_id}
        pair_gt_coo = None
        if mdict_pair_gt is not None:
            pair_gt_coo = mdict.to_sparse(np_cumsum, mdict_pair_gt)
            for index, matched in mdict_pair_gt.items():
                gt_pos = matched[0] + np_cumsum[*[index]]
                dict_gt.update(dict(zip(gt_pos[:, 0], matched[1])))
                dict_gt.update(dict(zip(gt_pos[:, 1], matched[1])))

            cm_gt = matplotlib.colormaps["prism"]  # type: ignore
            dict_color = dict(zip(dict_gt.values(), cm_gt([*dict_gt.values()])))
            dict_color_text_id: dict[int, str] = {i: ("black" if np.sum(v[:3] * [0.299, 0.587, 0.114]) > 0.45 else "white") for i, v in dict_color.items()}

            for index, gt_id in dict_gt.items():
                ax.plot(index, index, "o", color=dict_color[gt_id], markersize=15, zorder=7)  #! 在 dev 图的 中心线 上绘制全局 id 号
                ax.text(index, index, f"{gt_id}", fontsize=10, color=dict_color_text_id[gt_id], ha="center", va="center", zorder=8)

            if dict_edge is not None:
                # * 这里xy要互换一下，散点图的横轴是x轴，但是边索引关系的第0行是纵轴
                ax.scatter(np_edge[1], np_edge[0], marker=r"$e$", c="green", s=80, zorder=3)

        dict_result: dict[int, int] = {}  # *  dict{global_index: global_id}
        if mdict_pair is not None:
            curt_fig += 1
            ax_result = axes[curt_fig]
            Visiual_Tools._plot_block_line(ax_result, merge_aff.shape, np_cumsum)
            ax_result.xaxis.set_ticks_position("top")  # * 将x轴的位置设置在顶部
            ax_result.tick_params(axis="both")
            ax_result.invert_yaxis()  # * y轴反向
            if pair_gt_coo is not None:
                assert dict_gt is not None
                for index, gt_id in dict_gt.items():
                    ax_result.plot(index, index, "o", color=dict_color[gt_id], markersize=15, zorder=7)  #! 在 dev 图的 中心线 上绘制全局 id 号
                    ax_result.text(index, index, f"{gt_id}", fontsize=10, color=dict_color_text_id[gt_id], ha="center", va="center", zorder=8)
            pair_match_coo = mdict.to_sparse(np_cumsum, mdict_pair).tocoo()

            if mdict_pair_gt is not None:
                for i, j, v in zip(pair_match_coo.row, pair_match_coo.col, pair_match_coo.data):
                    # * 需要保证主对角线上的两个id都是相同的，这样的结果才是正确的,
                    # print(i, j)
                    if i in dict_gt and j in dict_gt:
                        if dict_gt[i] == dict_gt[j]:
                            #! 画不了emoji 提示没有找到这个字，即使采用了自己加载进来的emoji字体
                            # ax_result.text(j, i, r'😄', fontsize=15, fontname="Noto Color Emoji", ha='center', va='center', color="green")
                            ax_result.text(j, i, "✔", fontsize=20, ha="center", va="center", color="green")
                            continue
                    ax_result.text(j, i, "✖️", fontsize=18, ha="center", va="center", color="red")  #! 在 dev 图的 中心线 上绘制全局 id 号
            else:
                ax_result.plot(pair_match_coo.col, pair_match_coo.row, "o", color="black", markersize=15, zorder=7)  #! 在 dev 图的 中心线 上绘制全局 id 号

            fig.savefig("tmp/tmp.jpg")
            pass

        if mdict_pair_dev is not None:
            curt_fig += 1
            ax_dev = axes[curt_fig]
            list_xy = []
            list_aff = []
            for index, aff in mdict_pair_dev.items():
                index_xy = np.vstack(aff.nonzero())
                tmp = index_xy.T + np_cumsum[*[index]]
                list_xy += tmp.tolist()
                list_xy += tmp[:, ::-1].tolist()
                list_aff += 2 * aff[*index_xy].tolist()

            df_dev = DataFrame(np.array(list_xy), columns=["x", "y"])
            df_dev["value"] = np.array(list_aff)
            ax_dev = sns.scatterplot(
                data=df_dev,
                x="x",
                y="y",
                size="value",
                hue="value",
                ax=ax_dev,
                palette=cm_aff,
                sizes=scatter_sizes,
                size_norm=(0, 1),
                zorder=2,
            )
            #! 如果有gt，则在 dev 图的 中心线 上绘制全局 id 号
            if mdict_pair_gt is not None:
                assert dict_gt is not None
                for index, gt_id in dict_gt.items():
                    ax_dev.plot(index, index, "o", color=dict_color[gt_id], markersize=15, zorder=7)  #! 在 dev 图的 中心线 上绘制全局 id 号
                    ax_dev.text(index, index, f"{gt_id}", fontsize=10, color=dict_color_text_id[gt_id], ha="center", va="center", zorder=8)

            if dict_edge is not None:
                spare_edge_ij = sparse.coo_matrix((np.ones(np_edge.shape[1]), np_edge), shape=(sum_node, sum_node))
                edge_ij_double = ((spare_edge_ij + spare_edge_ij.T) == 2).tocoo()
                edge_ij_single = ((spare_edge_ij - spare_edge_ij.T) == 1).tocoo()
                # * 绘制初始边
                ax_dev.scatter(edge_ij_single.col, edge_ij_single.row, marker=r"$e$", c="g", s=50, zorder=3)  # * 画的小一点
                # * 绘制双边
                ax_dev.scatter(edge_ij_double.col, edge_ij_double.row, marker=r"$e$", c="c", s=50, zorder=3)  # * 画的小一点
                # * 绘制共轭边
                ax_dev.scatter(edge_ij_single.T.col, edge_ij_single.T.row, marker=r"$e$", c="blue", s=50, zorder=3)  # * 画的小一点

            Visiual_Tools._plot_block_line(ax_dev, merge_aff.shape, np_cumsum)
            ax_dev.xaxis.set_ticks_position("top")  # * 将x轴的位置设置在顶部
            ax_dev.tick_params(axis="both")
            ax_dev.invert_yaxis()  # * y轴反向
            sns.move_legend(ax_dev, "upper left", labelspacing=sum_node / 20, ncol=1, frameon=True, bbox_to_anchor=(1, 1), borderaxespad=0)

        sns.despine(fig=fig, left=True, bottom=True)  # * 去掉左边和下边的边框
        return fig

    @classmethod
    def draw_camera_connect_by_combination(
        cls,
        dict_data: dict[str, tuple[Image.Image, list[int]]],
        dict_tidx_ltrb: dict[int, np.ndarray],
        edge_index_pred: np.ndarray,  # 预测的关联
        dict_tidx_ctid: Optional[dict[int, int]] = None,
        edge_index_gt: Optional[np.ndarray] = None,  # 真值关联边
        dict_tuple_score: Optional[dict[tuple[int, int], float]] = None,  # 所有关连边的得分，以元组字典的形式索引
        linewidth_TP: int = 1,
        linewidth_FP: int = 1,
        linewidth_FN: int = 1,
        thresh_overlap: float = 0.2,  # * 考虑到目标可能是重合的，这里设置一个阈值，如果两个目标的iou大于这个阈值，则判断不是错误的连接
    ):
        """
        针对多个相机，用排列组合的方式显示两两目标的关联
        因为不是高性能需要，这里还是使用 matplotlib 进行绘制

        这里只接受 Image.Image 格式的输入，不再做其他适配

        Args:
            dict_data (dict[str, tuple[Image.Image, list[int]]]) : 图像数据字典，字典的键是相机名称，字典的值是一个元组，其中的元素分别是 :class:`Image.Image` 格式的图像, :class:`list` 的元素是 该相机内目标的 ``tidx``
            edge_index_pred (np.ndarray, optional) : 预测的关联边. Defaults to None.
            edge_index_gt (np.ndarray, optional) : 真值关联边. Defaults to None. 当该参数不为 None，则会自动判断预测边是否正确，正确的连接线使用 **绿色**，错误的连接线使用 **红色**，漏掉的连接线使用 **蓝色**
        会在该类中自动绘制锚框，因此需要传入的是一个没有锚框的原始图像
            pair_dataframe (DataFrame, optional) : 用于可视化的关联边. Defaults to None.
            `pair_dataframe.columns=['cid_a', 'cid_b', 'tidx_a', 'tidx_b', 'ctid_a', 'ctid_b', 'score', 'type', 'jid', 'gid']`
            dict_tidx_ctid (dict[int, int], optional) : 用于标记目标的 ctid 号. Defaults to None.
        example:
            .. code-block::

                cid_a  cid_b  tidx_a  tidx_b  ctid_a  ctid_b  score  type  jid  gid
                0       0      1       2       9       3       2  0.354  Sort   -1   -1
                1       0      1       3      10       4       3  0.995  Sort   -1   -1
                2       0      1       4      11       5       4  0.996  Sort   -1   -1
                3       0      2       0      13       1       1  0.862  Sort   -1   -1
                4       0      2       3      14       4       2  0.964  Sort   -1   -1
                5       0      2       4      15       5       3  1.000  Sort   -1   -1
                6       0      2       6      16       7       4  0.047  Sort   -1   -1
                7       0      2       7      20       8       8  0.000  Sort   -1   -1
                8       1      2       8      16       1       4  0.035  Sort   -1   -1
                9       1      2      10      14       3       2  0.996  Sort   -1   -1
                10      1      2      11      15       4       3  0.137  Sort   -1   -1
                11      1      2      12      20       5       8  1.000  Sort   -1   -1


        example:
            >>> dict_data = {
            'cam1': (img1, ltrb1),
            'cam2': (img2, ltrb2),
            'cam3': (img3, ltrb3),
            }
            # ltrb 是 np.ndarray[(n, 4), int]
            >>> edge_index_pred = np.array([[ 2,  3,  4,  0,  3,  4,  6,  7,  8, 10, 11, 12],
                                            [ 9, 10, 11, 13, 14, 15, 16, 20, 16, 14, 15, 20]])
            >>> edge_index_gt = np.array([[ 0,  0,  8,  2,  3,  3, 10,  4,  4, 11,  6, 12],
                                          [ 8, 13, 13,  9, 10, 14, 14, 11, 15, 15, 16, 20]])
            >>> fig = Pretty_Draw.draw_camera_connect_by_combination(
                      dict_data,
                      edge_index_pred=edge_index_keep,
                      edge_index_gt=debug_edge_index_gt,
                      )
        """

        # %%
        seq_img, seq_tidx = list(zip(*dict_data.values()))  # type: ignore
        seq_tidx: tuple[list[int]] = seq_tidx
        np_cumsum = np.cumsum([0] + [len(i) for i in seq_tidx])
        dict_tidx_xywh = {tidx: dict_convert_fn["ltrb"]["xywh"](ltrb) for tidx, ltrb in dict_tidx_ltrb.items()}
        n_t = len(dict_tidx_xywh)
        seq_name = [*dict_data.keys()]
        n_row = math.comb(len(seq_img), 2)

        # 分析真值
        if edge_index_gt is not None:
            edge_index_TP, edge_index_FN, edge_index_FP = match_result_check(edge_index_gt, edge_index_pred)

        # return None  # ? debug
        if dict_tuple_score is not None:
            np_score = np.array([*dict_tuple_score.values()])
            np_edge_index = np.array([*dict_tuple_score.keys()]).T
            max_ = max(np_edge_index.max(), np_edge_index.max()) + 1
            adj_score = sparse.coo_matrix((np_score, np_edge_index), shape=(max_, max_))
            adj_score = adj_score + adj_score.T

        # * 首先绘制锚框
        seq_img_with_bbox = []
        for name, (img, list_tidx) in dict_data.items():
            scale = img.width / 1920
            infos = [f"tidx:{i}" for i in list_tidx]
            ids = [dict_tidx_ctid[i] for i in list_tidx] if dict_tidx_ctid is not None else None
            ltrb = np.stack([dict_tidx_ltrb[i] for i in list_tidx])
            vis_img = Pretty_Draw.draw_bboxes(
                img, ltrb, ids=ids, outline=int(scale * 3), size_font=int(scale * 24), infos=infos, mask=0.2
            )  # * 在锚框的右上角显示 tidx 号， tidx 号的顺序由 dict_data 中给出
            seq_img_with_bbox.append(vis_img)

        # %% # * matplotlib figure
        fig, axs = plt.subplots(nrows=n_row, ncols=2, figsize=(25, 6 * n_row), dpi=Pretty_Draw.dpi)
        fig.subplots_adjust(wspace=0.1, hspace=0.2, top=0.95, bottom=0.05, left=0.05, right=0.90)
        ll_axs: list[list[Axes]] = np.array([axs]).tolist() if n_row == 1 else axs.tolist()

        # * 绘制组合图像
        for row, (i_img, j_img) in enumerate(itertools.combinations(range(len(seq_img)), 2)):
            axA = ll_axs[row][0]
            axB = ll_axs[row][1]
            axA.imshow(seq_img_with_bbox[i_img], zorder=1)
            axA.set_title(seq_name[i_img])
            axB.imshow(seq_img_with_bbox[j_img], zorder=1)
            axB.set_title(seq_name[j_img])

            graph_enhance.Utils_Edge.edge_index_filter_by_set(edge_index_TP, (set(seq_tidx[i_img]), set(seq_tidx[j_img])))
            edge_index_ij_TP = graph_enhance.Utils_Edge.edge_index_filter_by_set(edge_index_TP, (set(seq_tidx[i_img]), set(seq_tidx[j_img])))
            edge_index_ij_FP = graph_enhance.Utils_Edge.edge_index_filter_by_set(edge_index_FP, (set(seq_tidx[i_img]), set(seq_tidx[j_img])))
            edge_index_ij_FN = graph_enhance.Utils_Edge.edge_index_filter_by_set(edge_index_FN, (set(seq_tidx[i_img]), set(seq_tidx[j_img])))

            dict_score = None
            if dict_tuple_score is not None:
                score_ij_TP = np.array(adj_score[*edge_index_ij_TP]).reshape(-1) if edge_index_ij_TP.shape[1] != 0 else np.array([])
                score_ij_FP = np.array(adj_score[*edge_index_ij_FP]).reshape(-1) if edge_index_ij_FP.shape[1] != 0 else np.array([])
                score_ij_FN = np.array(adj_score[*edge_index_ij_FN]).reshape(-1) if edge_index_ij_FN.shape[1] != 0 else np.array([])
                dict_score = {"TP": score_ij_TP, "FP": score_ij_FP, "FN": score_ij_FN}

            matplotlib_misc.apply_connect_between_axes_TP_FP_FN(
                (axA, axB),
                {
                    "TP": (np_enhance.enhance_stack([dict_tidx_xywh[i][:2] for i in edge_index_ij_TP[0]]), np_enhance.enhance_stack([dict_tidx_xywh[i][:2] for i in edge_index_ij_TP[1]])),
                    "FP": (np_enhance.enhance_stack([dict_tidx_xywh[i][:2] for i in edge_index_ij_FP[0]]), np_enhance.enhance_stack([dict_tidx_xywh[i][:2] for i in edge_index_ij_FP[1]])),
                    "FN": (np_enhance.enhance_stack([dict_tidx_xywh[i][:2] for i in edge_index_ij_FN[0]]), np_enhance.enhance_stack([dict_tidx_xywh[i][:2] for i in edge_index_ij_FN[1]])),
                },
                dict_score=dict_score,  # type: ignore
                linewidth_FN=linewidth_FN,
                linewidth_TP=linewidth_TP,
                linewidth_FP=linewidth_FP,
            )
            # * 绘制分值表格
            scale = axB.dataLim.max[0] / 1920

            word = "TP:\n" + "\n".join([f"({i}, {j}) {score:0.3f}" for (i, j), score in zip(edge_index_ij_TP.T, dict_score["TP"])])
            axB.annotate(word, (axB.dataLim.max[0], axB.dataLim.max[1]), xytext=(axB.dataLim.max[0], axB.dataLim.max[1]))

            word = "FP:\n" + "\n".join([f"({i}, {j}) {score:0.3f}" for (i, j), score in zip(edge_index_ij_FP.T, dict_score["FP"])])
            axB.annotate(word, (axB.dataLim.max[0], axB.dataLim.max[1]), xytext=(axB.dataLim.max[0] + scale * 1.2 * Pretty_Draw.dpi, axB.dataLim.max[1]))

            word = "FN:\n" + "\n".join([f"({i}, {j}) {score:0.3f}" for (i, j), score in zip(edge_index_ij_FN.T, dict_score["FN"])])
            axB.annotate(word, (axB.dataLim.max[0], axB.dataLim.max[1]), xytext=(axB.dataLim.max[0] + scale * 2.4 * Pretty_Draw.dpi, axB.dataLim.max[1]))
            pass
        # plt.close("all")

        # fig.savefig("tmp/draw_camera_connect_by_combination.jpg")
        # %%
        return fig

    @classmethod
    def draw_edge_index(
        cls,
        edge_index,
        values=None,
        aux_linx_x: Optional[np.ndarray] = None,
        aux_linx_y: Optional[np.ndarray] = None,
        scatter_sizes=(5, 80),
        shape: Sequence[int] = None,
        size_norm: Tuple[int, int] = (0, 1),
        **kwargs: Unpack[Dict_Kwargs_scatter],
    ):
        """
        只绘制一张图，用以调试大矩阵中的邻接关系

        Args:
            edge_index : [description] 传入的邻接关系， **shape** 为 :code:`(2, edge_nums)`
            values (np.ndarray, torch.Tensor, optional): 值，应当和edge_index的输入顺序一致，可有可无，关系到邻接矩阵中显示的大小，如果为无，则所有标记大小一致

            aux_linx_x (np.ndarray, torch.Tensor, optional):
            aux_linx_y (np.ndarray, torch.Tensor, optional):
            shape (tuple or list, optional): 如果输入的是 稀疏矩阵 或 列表，这个 shape 则建议输入进来
            scatter_sizes: 散点图的大小范围

            fig_clean: True 表示使用的是 plt.figure(clean=True)，默认 False
        .. note::
            如果输入的不是 :class:`np.ndarray` 类型，则会将其判断为 :class:`torch.Tensor` 类型，并将其转化为 :class:`np.ndarray` 类型

        """
        assert edge_index.shape[0] == 2
        if values is not None:
            assert values.shape[0] == edge_index.shape[1]

        edge_index = convert_to_numpy(edge_index)
        values = convert_to_numpy(values)
        aux_linx_x = convert_to_numpy(aux_linx_x)
        aux_linx_y = convert_to_numpy(aux_linx_y)

        if shape is None:
            shape = (edge_index[0].max() + 1, edge_index[1].max() + 1)
        # ! 图像中的 x 和 y 与 i 和 j 是相反的，所以这里要进行转换
        cm_aff = kwargs.get("cm_aff", matplotlib.colormaps["inferno_r"])
        label_x = kwargs.get("label_x", "$j$")
        label_y = kwargs.get("label_y", "$i$")
        Oij = kwargs.get("Oij", None)  # *左上角偏移原点的位置，默认为 Oij=(x=0, y=0), x 表示左右，y 表示上下
        dpi = kwargs.get("dpi", Pretty_Draw.dpi)

        fig_clear = kwargs.get("fig_clear", False)

        # ! edge_index 也要换，因为第一行是 i，应当放在y上，所以这里要进行转换
        assert len(edge_index) == 2
        df_edge = DataFrame(edge_index[::-1].T, columns=[label_x, label_y])
        if values is not None:
            df_edge["value"] = values
        else:
            df_edge["value"] = 1  # * 全部设置默认的 0.5 的大小

        fig = plt.figure(figsize=(shape[1] / 3 + 1, shape[0] / 3), dpi=dpi, clear=fig_clear)

        hide_triu = kwargs.get("hide_triu", False)
        hide_tril = kwargs.get("hide_tril", False)
        hide_diag = kwargs.get("hide_diag", False)
        mid_label_x = kwargs.get("mid_label_x", None)
        mid_label_y = kwargs.get("mid_label_y", None)

        if hide_diag or mid_label_x:
            assert aux_linx_x is not None, "没有辅助线分不清哪个是对角线矩阵，也画不出分段中间标签"
            block_ptr_x = aux_linx_x.tolist()
            if aux_linx_x[-1] != shape[1]:
                block_ptr_x.append(shape[1])

        if hide_diag or mid_label_y:
            assert aux_linx_y is not None, "没有辅助线分不清哪个是对角线矩阵，也画不出分段中间标签"
            block_ptr_y = aux_linx_y.tolist()
            if aux_linx_y[-1] != shape[0]:
                block_ptr_y.append(shape[0])

        # ************* 遮挡修正 *************
        if hide_triu:
            df_edge = df_edge[df_edge[label_x] >= df_edge[label_y]]
        if hide_tril:
            df_edge = df_edge[df_edge[label_x] <= df_edge[label_y]]

        if hide_diag:
            assert (aux_linx_x is not None) and (aux_linx_y is not None), "没有辅助线分不清哪个是对角线矩阵"
            last_x, last_y = 0, 0
            for block_k in range(min(len(block_ptr_x), len(block_ptr_y))):
                max_x = block_ptr_x[block_k]
                max_y = block_ptr_y[block_k]
                pos_x = df_edge[label_x].to_numpy()
                pos_y = df_edge[label_y].to_numpy()
                flag_x = np.logical_and(last_x <= pos_x, pos_x < max_x)
                flag_y = np.logical_and(last_y <= pos_y, pos_y < max_y)
                flag = np.logical_and(flag_x, flag_y)
                df_edge = df_edge[~flag]
                last_x, last_y = max_x, max_y

        # ************* 画图 *************
        # * 如果绘制的最后一行最后一列正好就是矩阵的边界，则不绘制，但是如果调用了 hide_diag
        ax_edge = sns.scatterplot(
            data=df_edge,
            x=label_x,
            y=label_y,
            size="value",
            hue="value",
            ax=fig.gca(),
            sizes=scatter_sizes,
            size_norm=size_norm,
            zorder=2,
            palette=cm_aff,
        )
        Visiual_Tools._plot_block_line(
            ax_edge,
            shape,
            aux_linx_x=aux_linx_x,
            aux_linx_y=aux_linx_y,
            Oij=Oij,
        )
        ax_edge.invert_yaxis()  # * y轴反向
        ax_edge.tick_params(axis="both")
        ax_edge.spines["right"].set_visible(False)
        ax_edge.spines["top"].set_visible(False)
        ax_edge.spines["bottom"].set_visible(False)
        ax_edge.spines["left"].set_visible(False)
        ax_edge.xaxis.set_ticks_position("top")  # * 将x轴的位置设置在顶部
        ax_edge.yaxis.set_ticks_position("left")  # * 将y轴的位置设置在左侧
        ax_edge.xaxis.set_label_position("top")
        ax_edge.yaxis.set_label_position("left")
        # ************* 副标签 修正 比如对于分块图像，存放相机名称 *************
        if mid_label_x or mid_label_y:

            ax2: Axes = ax_edge._make_twin_axes()
            ax2.patch.set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            ax2.spines["bottom"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            ax2.xaxis.set_visible(False)
            ax2.yaxis.set_visible(False)
            if mid_label_x:
                ax2.xaxis.set_visible(True)
                ax2.xaxis.tick_bottom()
                mid_pos_x = (np.array([0] + block_ptr_x[:-1]) + block_ptr_x) / 2 - 0.5
                ax2.set_xticks(mid_pos_x)
                ax2.set_xticklabels(mid_label_x)
                ax2.set_xlim(ax_edge.get_xlim())  # * 否则可能会错位
                pass
            else:
                pass

            if mid_label_y:
                ax2.invert_yaxis()  # * y轴反向
                ax2.yaxis.set_visible(True)
                ax2.yaxis.tick_right()
                mid_pos_y = (np.array([0] + block_ptr_y[:-1]) + block_ptr_y) / 2 - 0.5
                ax2.set_yticks(mid_pos_y)
                ax2.set_yticklabels(mid_label_y)
                ax2.set_ylim(ax_edge.get_ylim())  # * 否则可能会错位
                pass
            else:
                pass

        # * 其他修正
        fig.subplots_adjust(right=0.8)
        # * 画图的设置
        # * frameon 图例的一个小灰盒子背景，图例放到左边
        # todo 经过不断更换 return 的放置位置，发现下面的这一行代码会造成内存溢出
        sns.move_legend(ax_edge, "upper left", labelspacing=shape[1] / 20, ncol=1, frameon=True, bbox_to_anchor=(1, 1), borderaxespad=5)

        return fig

    @classmethod
    def draw_adj(
        cls,
        adj,
        shape: Sequence[int] = None,
        **kwargs: Unpack[Dict_Kwargs_scatter],
    ) -> matplotlib.figure.Figure:
        """
        使用细节请看 :func:`Pretty_Draw.draw_edge_index`
        只适用于转换，将数据进行处理后，依旧调用的是 draw_edge_index
        Example:

        .. image:: http://pb.x-contion.top/wiki/2023_08/15/3_vis_adj_sum.png
            :width: 800px
            :align: center

        Args:
            adj : The adjacency matrix. 可以是列表，这样会作为主对角线上的方阵进行处理
            shape (tuple or list, optional):
                如果输入的是 稀疏矩阵 或 列表，这个 shape 则建议输入进来. 默认使用 ``adj`` 的尺寸
        .. note::
            返回的 Figure 有两个 ax，请在 np_axes[0] 中继续绘制，而不是 fig.gca()
        """
        if isinstance(adj, list):
            # * 如果是列表，必然通过稀疏矩阵的方式进行可视化
            from pretty_tools.datastruct import np_enhance

            edge_index, values = np_enhance.convert_listadj_to_edgevalue(adj)

        elif isinstance(adj, (sparse.sparray, sparse.spmatrix)):
            adj = convert_to_numpy(adj, sparse_shape=shape)
            adj = adj.tocsr()
            edge_index = np.stack(adj.nonzero())
            values = np.array(adj[*edge_index]).reshape(-1)
        elif isinstance(adj, np.ndarray):
            adj = convert_to_numpy(adj)
            edge_index = np.indices(adj.shape).reshape(2, -1)
            values = np.array(adj).flatten()
        else:
            adj = convert_to_numpy(adj, sparse_shape=shape)
            edge_index = np.stack(adj.nonzero())
            values = adj[*edge_index]

        if shape is None:
            shape = adj.shape  # type: ignore

        fig = cls.draw_edge_index(
            edge_index,
            values,
            shape=shape,
            **kwargs,
        )
        return fig

    @classmethod
    def visual_tensor_heatmap(cls, tensor, as_img=False, annot=True):
        """可视化张量，有若干个模式可以选择


        Args:
            tensor (torch.Tensor): 输入的张量
            as_img (bool, optional): 是否将其绘制成图像

        .. note::
            默认张量的颜色通道在前，即 [C, N, M]


        """
        if isinstance(tensor, list):
            from pretty_tools.datastruct import np_enhance

            edge_index, values = np_enhance.convert_listadj_to_edgevalue(tensor)
            list_shape = [t.shape for t in tensor]
            total_shape = np.sum(list_shape, axis=0)
            # * 转换成稠密矩阵（这一步没必要优化了，毕竟画出来的图肯定比这个的稠密矩阵还大）
            adj = sparse.coo_matrix((values, (edge_index[0], edge_index[1])), shape=total_shape).todense()
            tensor = torch.from_numpy(adj)

        assert isinstance(tensor, torch.Tensor)
        tensor = tensor.detach().cpu()
        if tensor.shape[0] > 100 or tensor.shape[1] > 100:
            if not as_img:
                raise UserWarning(f"输入的张量过大(shape={[*tensor.shape]} 某个维度大于 100)，未指定 as_img 可能会导致内存占用过大")

            pass
        if tensor.ndim == 2:
            fig_size = tensor.shape[::-1]
        else:
            fig_size = tensor.shape[-2:][::-1]

        if as_img:
            fig_size = tuple(np.array([*fig_size]) / 8)  # * 这里是一个调整大小的参数，可以根据需要进行调整
        else:
            if annot:
                fig_size = tuple(np.array([*fig_size]))  # * 这里是一个调整大小的参数，可以根据需要进行调整
            else:
                fig_size = tuple(np.array([*fig_size]) / 2)  # * 这里是一个调整大小的参数，可以根据需要进行调整

        fig = plt.figure(figsize=fig_size, dpi=Pretty_Draw.dpi)

        if tensor.min() < 0:
            cm = matplotlib.colormaps["bwr"]  # type: ignore
        else:
            cm = matplotlib.colormaps["jet"]  # type: ignore

        if as_img:
            ax = sns.heatmap(tensor, center=0, ax=fig.gca(), cmap=cm, square=True, annot=False)
            ax.xaxis.set_ticks_position("top")  # * 将x轴的位置设置在顶部
            labels = ax.get_xticklabels()
            ax.set_xticklabels(labels=labels, rotation=90)
        else:
            ax = sns.heatmap(tensor, center=0, ax=fig.gca(), cmap=cm, square=True, annot=annot, fmt=".2f")
            ax.xaxis.set_ticks_position("top")  # * 将x轴的位置设置在顶部

        return fig

    @classmethod
    def draw_PR_curve(cls, P, R, thresh, cmap="turbo"):
        """
        绘制PR曲线

        Args:
            P (np.ndarray): precision
            R (np.ndarray): recall
            thresh (np.ndarray): 阈值

        #todo F1 的值还没有在图上进行说明
        """
        from matplotlib.collections import LineCollection
        from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

        fig = plt.figure(figsize=(8, 7), dpi=3 * Pretty_Draw.dpi)
        ax = fig.gca()

        # * P-R 曲线，R 应当是 x 轴，P 应当是 y 轴
        points = np.array([R, P]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)  # * 有几个点就切分成几个段，这样每个段都能有一个颜色
        F1 = 2 * (R * P) / (R + P + 1e-9)
        norm = Normalize(F1.min(), F1.max())
        lc = LineCollection(segments, cmap=cmap, norm=norm)  # type: ignore
        lc.set_array(F1)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)  # type: ignore
        fig.colorbar(line)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        # fig.savefig("tmp/tmp.jpg")  #? debug
        # plt.close('all')  #? debug
        return fig

    @classmethod
    def draw_heatmaps(
        cls,
        nrows: int,
        ncols: int,
        tuple_matrix: Union[Sequence[Sequence[np.ndarray]], Sequence[np.ndarray]],
        figsize: Optional[tuple[int, int]] = None,
        dpi: int = None,
        annot: bool = False,
        square: bool = False,
        constrained_layout=True,
        cmap=None,
        fmt: str = ".2g",
        use_same_cbar: bool = True,
        cbar_kws: Optional[dict] = None,
        kwargs_grid: Dict_Kwargs_GridSpec = None,
    ) -> tuple[Figure, np.ndarray[Any, Axes]]:
        """
        同时绘制多个热图 (基于 seaborn)。并且共享同一个颜色轴，通过调用 ``GridSpec`` 解决了共享颜色轴时创建新轴进而影响了其他轴显示效果的问题

        Args:
            nrows (int): 行数
            ncols (int): 列数
            tuple_matrix : 矩阵形式 或者 :class:`tuple` 形式存放的 :class:`np.ndarray` 矩阵，用以可视化
            use_same_cbar: 是否共用颜色条
            cbar_kws: 颜色条的参数，可选. 如 cbar_kws={"shrink": .82} 对颜色条进行缩放
            kwargs_grid: Gridspec 的参数
            constrained_layout: 是否使用 constrained_layout 使得各子图之间的距离自动调整

        .. note::

            默认将 x 轴的刻度绘制在图像的上方

        .. todo:

            #todo: 不知道为什么，画出来的颜色条有点过长

        Example
        -------

        .. code-block:: python

            import numpy as np
            from pretty_tools.visualization.draw import Pretty_Draw, Visiual_Tools

            np_rand0 = np.arange(8 * 8)[::-1].reshape(8, 8)
            np_rand1 = np.arange(8 * 8).reshape(8, 8).T
            np_rand2 = np.arange(8 * 8)[::-1].reshape(8, 8).T
            np_rand3 = np.arange(8 * 8).reshape(8, 8)
            fig, axes = Pretty_Draw.draw_heatmaps(2, 2, ((np_rand0, np_rand1), (np_rand2, np_rand3)), figsize=(10, 10), square=True)
            axes[0, 0].set_title("np_rand0")
            axes[0, 1].set_title("np_rand1")
            axes[1, 1].set_title("np_rand3")
            axes[1, 0].set_title("np_rand2")

            visual_att_img = Visiual_Tools.fig_to_image(fig) # 把图像转换成 np.ndarray
            fig.savefig("tmp/demo.png") #保存图像

        Example
        -------

        .. code-block:: python

            fig, _ = Pretty_Draw.draw_heatmaps(1, 1, (matrix_iou,), square=True, annot=True)
            fig.savefig("tmp/matrix_iou.png")

        .. image:: http://pb.x-contion.top/wiki/2023_09/15/3_demo.png
            :alt: draw_heatmaps_demo
            :width: 500px
            :height: 500px


        Example
        -------
        .. image:: http://pb.x-contion.top/wiki/2023_09/15/3_attention_coefficient.png
            :alt: attention_coefficient
            :width: 500px
            :height: 270px

        """
        import matplotlib.cm as cm
        from seaborn import cm as sns_cm

        if cmap is None:
            cmap = sns_cm.rocket
        elif isinstance(cmap, str):
            cmap = matplotlib.colormaps[cmap]  # type: ignore
        if cbar_kws is None:
            cbar_kws = {}

        if dpi is None:
            dpi = cls.dpi
        assert isinstance(nrows, int)
        assert isinstance(ncols, int)

        if nrows == 1:
            tuple_matrix = [tuple_matrix]  # type: ignore

        if figsize is None:
            figsize = (ncols * 5, nrows * 5)  # * figsize的顺序是 宽高，而不是高宽

        assert len(tuple_matrix) == nrows, "传入的可视化矩阵应当是 元组形式，元组的尺寸应当为 (nrows, ncols) "
        assert len(tuple_matrix[0]) == ncols, "传入的可视化矩阵应当是 元组形式，元组的尺寸应当为 (nrows, ncols) "

        fig = plt.figure(dpi=dpi, figsize=figsize, constrained_layout=constrained_layout)  # * 使得各子图之间的距离自动调整

        # fig = plt.figure(dpi=dpi, figsize=figsize)
        # * 本质上分了两个区域，一个是热图，一个是颜色条，颜色条这里只能放在右边(也可以改)
        if use_same_cbar:
            ori_kwargs_grid = {"width_ratios": [1] * ncols + [0.1], "figure": fig}
            if kwargs_grid is not None:
                ori_kwargs_grid.update(kwargs_grid)

            gs = GridSpec(nrows, ncols + 1, **ori_kwargs_grid)
            vmax = max([adj.max() for adj in itertools.chain(*tuple_matrix)])
            vmin = max([adj.min() for adj in itertools.chain(*tuple_matrix)])
        else:
            index = 0
            pass
            # gs = GridSpec(nrows, ncols, width_ratios=[1] * ncols, height_ratios=[1] * nrows, figure=fig)

        np_ax = np.zeros((nrows, ncols), dtype=np.object_)

        wrap_heatmap_fn = functools.partial(sns.heatmap, cmap=cmap, annot=annot, square=square, fmt=fmt)

        for i in range(nrows):
            for j in range(ncols):

                adj = tuple_matrix[i][j]

                if use_same_cbar:
                    ax = fig.add_subplot(gs[i, j])
                    wrap_heatmap_fn(adj, ax=ax, cbar=False, vmin=vmin, vmax=vmax)  # * 共用颜色条
                else:
                    index += 1
                    ax = fig.add_subplot(nrows, ncols, index)
                    wrap_heatmap_fn(adj, ax=ax, cbar_kws=cbar_kws)  # * 不共用颜色条

                # ? debug ===========
                # axpos = ax.get_position()
                # rect = matplotlib.patches.Rectangle((axpos.x0, axpos.y0), axpos.width, axpos.height, lw=3, ls="--", ec="r", fc="none", alpha=0.5, transform=ax.figure.transFigure)
                # ax.add_patch(rect)
                # ax.add_artist(rect)
                # ? debug ===========
                ax.xaxis.set_ticks_position("top")  # ! 默认将 x 轴的位置设置在顶部
                np_ax[i, j] = ax

        if use_same_cbar:
            import matplotlib.colors as mcolors

            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            c_ax = fig.add_subplot(gs[:, -1])
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=c_ax)  # colorbar(ax=c_ax) 和  colorbar(cax=c_ax) 是有区别的，前者会绘制颜色条，但是左侧留空了
        else:
            pass

        return fig, np_ax


def add_right_cax(ax, pad: int, width: int):
    """
    在一个ax右边追加与之等高的cax.

    pad是cax与ax的间距,width是cax的宽度.
    """
    axpos = ax.get_position()
    caxpos = mtransforms.Bbox.from_extents(axpos.x1 + pad, axpos.y0, axpos.x1 + pad + width, axpos.y1)
    cax = ax.figure.add_axes(caxpos)

    return cax
