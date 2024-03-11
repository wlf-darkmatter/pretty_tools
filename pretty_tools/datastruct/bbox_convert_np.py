"""

默认不加载Torch，以提高运行效率
但是支持转换 torch.Tensor 的数据

"""

from typing import Any, Callable, Dict, List, Literal, Tuple, Union, TypeVar
from copy import deepcopy
import numpy as np

Item_Bbox = Literal["ltrb", "ltwh", "xywh"]

T = TypeVar("T")


def __quick_convert(bbox: T, inplace: bool) -> tuple[bool, T]:
    assert hasattr(bbox, "shape"), "输入的参数必须是 torch.Tensor 或者 numpy.ndarray"
    if not inplace:
        bbox = deepcopy(bbox)
    if len(bbox.shape) == 2:
        return False, bbox
    else:
        if bbox.__class__.__name__ == "Tensor":
            bbox = bbox.unsqueeze(0)
        elif bbox.__class__.__name__ == "ndarray":
            bbox = np.array([bbox])
        else:
            raise TypeError("输入的参数必须是 torch.Tensor 或者 numpy.ndarray")
        return True, bbox


def ltrb_to_ltwh(ltrb: T, inplace=False) -> T:
    ndim1, ltwh = __quick_convert(ltrb, inplace)

    ltwh[:, 2] = ltwh[:, 2] - ltwh[:, 0]
    ltwh[:, 3] = ltwh[:, 3] - ltwh[:, 1]
    if ndim1:
        return ltwh[0]
    return ltwh


def ltwh_to_ltrb(ltwh: T, inplace=False) -> T:
    ndim1, ltrb = __quick_convert(ltwh, inplace)

    ltrb[:, 2] = ltrb[:, 2] + ltrb[:, 0]
    ltrb[:, 3] = ltrb[:, 3] + ltrb[:, 1]
    if ndim1:
        return ltrb[0]
    return ltrb


def ltwh_to_xywh(ltwh: T, inplace=False) -> T:
    """
    左上宽高 转 中心宽高
    """
    ndim1, xywh = __quick_convert(ltwh, inplace)

    xywh[:, 0] = xywh[:, 0] + xywh[:, 2] / 2.0
    xywh[:, 1] = xywh[:, 1] + xywh[:, 3] / 2.0
    if ndim1:
        return xywh[0]
    return xywh


def xywh_to_ltwh(xywh: T, inplace=False) -> T:
    """ """
    ndim1, ltwh = __quick_convert(xywh, inplace)

    ltwh[:, 0] = ltwh[:, 0] - ltwh[:, 2] / 2.0  # L
    ltwh[:, 1] = ltwh[:, 1] - ltwh[:, 3] / 2.0  # T
    if ndim1:
        return ltwh[0]
    return ltwh


def xywh_to_ltrb(xywh: T, inplace=False) -> T:
    """ """
    ndim1, ltrb = __quick_convert(xywh, inplace)

    ltrb[:, 0] = ltrb[:, 0] - ltrb[:, 2] / 2.0  # L
    ltrb[:, 1] = ltrb[:, 1] - ltrb[:, 3] / 2.0  # T
    ltrb[:, 2] = ltrb[:, 2] + ltrb[:, 0]
    ltrb[:, 3] = ltrb[:, 3] + ltrb[:, 1]
    if ndim1:
        return ltrb[0]
    return ltrb


def ltrb_to_xywh(ltrb: T, inplace=False) -> T:
    ndim1, xywh = __quick_convert(ltrb, inplace)

    xywh[:, 2] = xywh[:, 2] - xywh[:, 0]  # W = R - L  # LTWB
    xywh[:, 3] = xywh[:, 3] - xywh[:, 1]  # H = B - T  # LTWH
    xywh[:, 0] = xywh[:, 0] + xywh[:, 2] / 2.0  # X = L + W / 2 # XTWH
    xywh[:, 1] = xywh[:, 1] + xywh[:, 3] / 2.0  # Y = B + H / 2 # XYWH
    if ndim1:
        return xywh[0]
    return xywh


dict_convert_fn = {
    "ltrb": {
        "ltwh": ltrb_to_ltwh,
        "xywh": ltrb_to_xywh,
        "ltrb": lambda x: x,
    },
    "ltwh": {
        "ltrb": ltwh_to_ltrb,
        "xywh": ltwh_to_xywh,
        "ltwh": lambda x: x,
    },
    "xywh": {
        "ltrb": xywh_to_ltrb,
        "ltwh": xywh_to_ltwh,
        "xywh": lambda x: x,
    },
    ("ltrb", "ltwh"): ltrb_to_ltwh,
    ("ltrb", "xywh"): ltrb_to_xywh,
    ("ltrb", "ltrb"): lambda x: x,
    ("ltwh", "ltrb"): ltwh_to_ltrb,
    ("ltwh", "xywh"): ltwh_to_xywh,
    ("ltwh", "ltwh"): lambda x: x,
    ("xywh", "ltrb"): xywh_to_ltrb,
    ("xywh", "ltwh"): xywh_to_ltwh,
    ("xywh", "xywh"): lambda x: x,
}
