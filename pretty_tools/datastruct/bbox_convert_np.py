from typing import Any, Callable, Dict, List, Tuple, Literal, Union

import numpy as np

Item_Bbox = Literal["ltrb", "ltwh", "xywh"]


def ltrb_to_ltwh(ltrb):
    ltwh = np.array(ltrb)
    ndim1 = ltwh.ndim == 1
    if ndim1 == 1:
        ltwh = np.array([ltwh])
    ltwh[:, 2] -= ltwh[:, 0]
    ltwh[:, 3] -= ltwh[:, 1]
    if ndim1:
        return ltwh[0]
    return ltwh


def ltwh_to_ltrb(ltwh):
    ltrb = np.array(ltwh)
    ndim1 = ltrb.ndim == 1
    if ndim1 == 1:
        ltrb = np.array([ltrb])
    ltrb[:, 2] += ltrb[:, 0]
    ltrb[:, 3] += ltrb[:, 1]
    if ndim1:
        return ltrb[0]
    return ltrb


def ltwh_to_xywh(ltwh):
    """
    左上宽高 转 中心宽高
    """
    xywh = np.array(ltwh)
    ndim1 = xywh.ndim == 1
    if ndim1 == 1:
        xywh = np.array([xywh])
    xywh[:, 0] += xywh[:, 2] / 2.0
    xywh[:, 1] += xywh[:, 3] / 2.0
    if ndim1:
        return xywh[0]
    return xywh


def xywh_to_ltwh(xywh):
    """ """
    ltwh = np.array(xywh)
    ndim1 = ltwh.ndim == 1
    if ndim1 == 1:
        ltwh = np.array([ltwh])
    ltwh[:, 0] -= ltwh[:, 2] / 2.0  # L
    ltwh[:, 1] -= ltwh[:, 3] / 2.0  # T
    if ndim1:
        return ltwh[0]
    return ltwh


def xywh_to_ltrb(xywh):
    """ """
    ltrb = np.array(xywh)
    ndim1 = ltrb.ndim == 1
    if ndim1:
        ltrb = np.array([ltrb])
    ltrb[:, 0] -= ltrb[:, 2] / 2.0  # L
    ltrb[:, 1] -= ltrb[:, 3] / 2.0  # T
    ltrb[:, 2] += ltrb[:, 0]
    ltrb[:, 3] += ltrb[:, 1]
    if ndim1:
        return ltrb[0]
    return ltrb


def ltrb_to_xywh(ltrb):
    xywh = np.array(ltrb)
    ndim1 = xywh.ndim == 1
    if ndim1 == 1:
        xywh = np.array([xywh])
    xywh[:, 2] -= xywh[:, 0]  # W = R - L  # LTWB
    xywh[:, 3] -= xywh[:, 1]  # H = B - T  # LTWH
    xywh[:, 0] += xywh[:, 2] / 2.0  # X = L + W / 2 # XTWH
    xywh[:, 1] += xywh[:, 3] / 2.0  # Y = B + H / 2 # XYWH
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
