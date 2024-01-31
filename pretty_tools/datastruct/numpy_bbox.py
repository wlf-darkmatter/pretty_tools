from typing import Optional, Union

import numpy as np
import shapely
from pretty_tools.datastruct.cython_bbox import cy_bbox_overlaps_flag
from shapely.geometry import Polygon


def loop_union(a, b: Optional[Union[list, Polygon]] = None) -> Polygon:
    if b is None:
        assert isinstance(a, list)
        b = a[1:]
        a = a[0]
    if not isinstance(b, list):
        b = [b]
    for i in b:  # type: ignore
        a = a.union(i)
    return a


def loop_intersection(a, b: Optional[Union[list, Polygon]] = None) -> Polygon:
    if b is None:
        assert isinstance(a, list)
        b = a[1:]
        a = a[0]
    if not isinstance(b, list):
        b = [b]
    for i in b:  # type: ignore
        a = a.intersection(i)
    return a


def loop_difference(a, b: Optional[Union[list, Polygon]] = None) -> Polygon:
    if b is None:
        assert isinstance(a, list)
        b = a[1:]
        a = a[0]
    if not isinstance(b, list):
        b = [b]
    for i in b:  # type: ignore
        a = a.difference(i)
    return a


def bbox_no_overlaps_area(boxes: np.ndarray, query_boxes: np.ndarray, ignore_self=True, ratio=False) -> np.ndarray:
    """
    Args:
        boxes (np.ndarray): (N, 4) ndarray of float, ltrb format
        query_boxes (np.ndarray): (K, 4) ndarray of float, ltrb format
        ignore_self (bool): if True, ignore self overlaps, default True, required len(boxes) == len(query_boxes)
        ratio (bool): if True, return area ratio
    Return:
        no_overlaps_area (np.ndarray): boxes 锚框中 未被 query_boxes 所有锚框遮挡的区域面积
    """

    N = boxes.shape[0]
    K = query_boxes.shape[0]
    if N == 0:
        return np.array([])
    if K == 0:
        if ratio:
            return np.ones((N))
        else:
            return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # * 改一改其实能拓展到多边形
    pass
    overlap_flag = cy_bbox_overlaps_flag(boxes, query_boxes)
    if ignore_self:
        assert N == K, "N!= K"
        overlap_flag[np.arange(N), np.arange(N)] = False
    # empty_ploy = Polygon([(0, 0), (0, 0), (0, 0), (0, 0)])  # * 作为一个主体，方便代码的编写
    list_poly = np.array([Polygon([(l, t), (r, t), (r, b), (l, b)]) for l, t, r, b in boxes])
    list_query_poly = np.array([Polygon([(l, t), (r, t), (r, b), (l, b)]) for l, t, r, b in query_boxes])

    list_no_overlaps_area = []
    for i, list_flag in enumerate(overlap_flag):
        self_poly: Polygon = list_poly[i]
        if list_flag.any():  # * 说明有别的框跟这个重叠
            list_with_overlap = list_query_poly[list_flag].tolist()
            merge_poly = loop_union(list_with_overlap)
            diff: Polygon = self_poly.difference(merge_poly)
            if ratio:
                no_overlaps_area = diff.area / self_poly.area
            else:
                no_overlaps_area = diff.area
        else:
            if ratio:
                no_overlaps_area = 1
            else:
                no_overlaps_area = self_poly.area
        list_no_overlaps_area.append(no_overlaps_area)

    return np.array(list_no_overlaps_area)


if __name__ == "__main__":
    np_bboxes = np.array(
        [
            [0, 0, 4, 5],
            [2, 2, 8, 8],
            [2, 2, 9, 9],
            [10, 0, 15, 10],
            [0, 0, 3, 7],
        ],
        dtype=np.float64,
    )

    bbox_no_overlaps_area(np_bboxes, np_bboxes)

    pass
