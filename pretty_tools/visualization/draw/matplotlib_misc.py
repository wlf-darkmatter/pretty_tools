"""
这里存放了一些常用的调用函数，专门提供给 matplotlib 使用
"""
import random
from copy import deepcopy
from typing import Iterable, Literal, Optional

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
from matplotlib.transforms import Affine2D, offset_copy

Type_Connect = Literal["unique", "TP", "FP", "FN"]


def apply_connect_between_axes_TP_FP_FN(
    pair_axes,
    dict_point: dict[Type_Connect, tuple[np.ndarray, np.ndarray]],
    dict_score: Optional[dict[Type_Connect, np.ndarray]] = None,
    zorder=3,
    linewidth_TP=1,
    linewidth_FP=1,
    linewidth_FN=1,
):
    """
    这个主要是再次封装，并且统一连接线的弧度

    Args:
        pair_axes (tuple[plt.Axes, plt.Axes]): 两个 axes
    """

    merge_point1, _ = list(zip(*dict_point.values()))
    merge_point1 = np.concatenate([i for i in merge_point1 if len(i) != 0])
    _max = merge_point1.max(axis=0)
    _min = merge_point1.min(axis=0)
    axesA, axesB = pair_axes
    fn = lambda xy: np.prod([-1, 1] * (xy - _min) / (_max - _min) + [1, -0.5], axis=1)

    list_color = []
    list_score = []
    for connect_type, (np_points_1, np_points_2) in dict_point.items():
        if connect_type == "FN":
            np_color = np.repeat(np.array([0, 0, 1]), len(np_points_1)).reshape(3, -1).T
            np_linewidth = np.array([linewidth_FN for _ in range(len(np_points_1))])
        elif connect_type == "TP":
            np_color = np.repeat(np.array([0, 1, 0]), len(np_points_1)).reshape(3, -1).T
            np_linewidth = np.array([linewidth_TP for _ in range(len(np_points_1))])
        elif connect_type == "FP":
            np_color = np.repeat(np.array([1, 0, 0]), len(np_points_1)).reshape(3, -1).T
            np_linewidth = np.array([linewidth_FP for _ in range(len(np_points_1))])
        np_score = dict_score[connect_type] if dict_score is not None else None
        list_color.append(np_color)
        list_score.append(np_score)
        if len(np_points_1) == len(np_points_2) == 0:
            continue
        apply_connect_between_axes(
            axesA,
            axesB,
            np_points_1,
            np_points_2,
            np_color=np_color,
            np_score=np_score,
            np_rad=fn(np_points_1),
            np_linewidth=np_linewidth,
            zorder=zorder,
        )
        pass
    # todo np_score 目前不好用上


def apply_connect_between_axes(
    axesA,
    axesB,
    np_points_1: np.ndarray,
    np_points_2: np.ndarray,
    np_color: Optional[np.ndarray] = None,
    np_score: Optional[np.ndarray] = None,
    np_rad: Optional[np.ndarray] = None,  # type: ignore
    np_linewidth: Optional[np.ndarray] = None,
    zorder=3,
):
    """
    Args:

        np_color (np.ndarray): 一个颜色数组，限定尺寸格式为 (n, 3)，值域为 [0, 1]

    弧度范围是 (-0.3, 0.3) 具体根据左图的位置来调整
    """

    assert len(np_points_1) == len(np_points_2)
    if len(np_points_1) == 0:
        return
    if np_rad is None:
        y_norm = deepcopy(np_points_1[:, 1])
        y_norm -= y_norm.min()
        y_norm /= y_norm.max()

        x_norm = deepcopy(np_points_1[:, 0])
        x_norm -= x_norm.min()
        x_norm /= x_norm.max()
        np_rad: Iterable[np.ndarray] = 0.4 * (1 - x_norm) * (y_norm - 0.5)

    if np_color is None:
        np_color = np.array([(random.random(), random.random(), random.random()) for _ in range(len(np_points_1))])
    if np_linewidth is None:
        np_linewidth = np.array([1 for _ in range(len(np_points_1))])

    for point_1, point_2, rad, color, linewidth in zip(np_points_1, np_points_2, np_rad, np_color, np_linewidth):
        con = ConnectionPatch(
            xyA=point_1,
            xyB=point_2,
            coordsA="data",
            coordsB="data",
            axesA=axesA,
            axesB=axesB,
            arrowstyle="->",
            color=color,
            connectionstyle=f"arc3,rad={rad:.04f}",  # (-0.2,0.2) 的一个随机弧度
            zorder=zorder,
            linewidth=linewidth,
            # path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()],
        )

        axesB.add_artist(con)  # * 不能放置在 axesA 上，否则连接线在 axesA 上会绘制，但是在 axesB 上会被图像给覆盖
        # * 在两个图中绘制一个圆点
        axesA.plot(*point_1, "ro", markersize=4, zorder=zorder)
        axesB.plot(*point_2, "ro", markersize=4, zorder=zorder)
    # axesA.figure.savefig("tmp/tmp.jpg")


def rainbow_text(x, y, strings, colors, orientation="horizontal", ax=None, **kwargs):
    """
    https://matplotlib.org/stable/gallery/text_labels_and_annotations/rainbow_text.html

    Take a list of *strings* and *colors* and place them next to each
    other, with text strings[i] being shown in colors[i].

    Parameters
    ----------
    x, y : float
        Text position in data coordinates.
    strings : list of str
        The strings to draw.
    colors : list of color
        The colors to use.
    orientation : {'horizontal', 'vertical'}
    ax : Axes, optional
        The Axes to draw into. If None, the current axes will be used.
    **kwargs
        All other keyword arguments are passed to plt.text(), so you can
        set the font size, family, etc.
    """
    if ax is None:
        ax = plt.gca()
    t = ax.transData
    fig = ax.figure
    canvas = fig.canvas

    assert orientation in ["horizontal", "vertical"]
    if orientation == "vertical":
        kwargs.update(rotation=90, verticalalignment="bottom")

    for s, c in zip(strings, colors):
        text = ax.text(x, y, s + " ", color=c, transform=t, **kwargs)

        # Need to draw to update the text position.
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        # Convert window extent from pixels to inches
        # to avoid issues displaying at different dpi
        ex = fig.dpi_scale_trans.inverted().transform_bbox(ex)

        if orientation == "horizontal":
            t = text.get_transform() + offset_copy(Affine2D(), fig=fig, x=ex.width, y=0)
        else:
            t = text.get_transform() + offset_copy(Affine2D(), fig=fig, x=0, y=ex.height)
