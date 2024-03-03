# --------------------------------------------------------
# ContionTrack
# Copyright (c) 2023 BIT
# Licensed under The MIT License [see LICENSE for details]
# Written by Wang Lingfeng
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Sergey Karayev
# --------------------------------------------------------
# cython:language_level=3

"""
Numpy中的 ” float64″ 类型对应 Cython中 的 ”double” 类型，”int32″ 类型对应 ”int”类型。

"""


cimport cython

import numpy as np

cimport numpy as cnp
from cpython cimport bool


def cy_bbox_overlaps_iou(
    cnp.ndarray[cnp.float_t, ndim=2] boxes,
    cnp.ndarray[cnp.float_t, ndim=2] query_boxes
    ):
    """
    Args:
        boxes (np.ndarray): (N, 4) ndarray of float, ltrb format
        query_boxes (np.ndarray): (K, 4) ndarray of float, ltrb format

    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes


    .. note::
        因为算法中对像素点的边界进行了+1 的操作，导致无法处理纯比例构成的 IoU 的占比大小。

    """
    cdef unsigned int N = boxes.shape[0] #type: ignore
    cdef unsigned int K = query_boxes.shape[0] #type: ignore
    cdef cnp.ndarray[cnp.float_t, ndim=2] overlaps = np.zeros((N, K), dtype=float) #type: ignore
    cdef cnp.float_t iw, ih, box_area
    cdef cnp.float_t ua
    cdef unsigned int k, n
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1)
                if ih > 0:
                    ua = float((boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1) + box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def cy_bbox_overlaps_area(
    cnp.ndarray[cnp.float_t, ndim=2] boxes,
    cnp.ndarray[cnp.float_t, ndim=2] query_boxes
    ):
    """
    返回的是交集 的绝对面积

    Parameters
    ----------
    boxes: (N, 4) ndarray of float, ltrb format
    query_boxes: (K, 4) ndarray of float, ltrb format
    Returns
    -------
    overlaps_intersection: (N, K) ndarray of `overlaps_intersection = overlap / area(boxes)`
    """
    cdef unsigned int N = boxes.shape[0] #type: ignore
    cdef unsigned int K = query_boxes.shape[0] #type: ignore
    cdef cnp.ndarray[cnp.float_t, ndim=2] overlaps_area = np.zeros((N, K), dtype=float) #type: ignore
    cdef cnp.float_t iw, ih

    cdef unsigned int k, n
    for n in range(N):
        for k in range(K):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1)
                if ih > 0:
                    overlaps_area[n, k] = iw * ih

    return overlaps_area


def cy_bbox_overlaps_flag(
    cnp.ndarray[cnp.float_t, ndim=2] boxes,
    cnp.ndarray[cnp.float_t, ndim=2] query_boxes
    ):
    """
    输入必须是 浮点数
    """
    cdef unsigned int N = boxes.shape[0] #type: ignore
    cdef unsigned int K = query_boxes.shape[0] #type: ignore
    cdef cnp.ndarray[cnp.npy_bool, ndim=2] overlaps_flag = np.zeros((N, K), dtype=np.bool_) #type: ignore
    cdef cnp.float_t iw, ih
    cdef unsigned int k, n

    for n in range(N):
        for k in range(K):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1)
                if ih > 0:
                    overlaps_flag[n, k] = True

    return overlaps_flag

