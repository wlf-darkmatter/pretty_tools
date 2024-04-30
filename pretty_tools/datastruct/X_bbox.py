from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Tuple, TypeVar, Union

import numpy as np
import torch
from torch import Tensor
from torchvision.ops import box_area
from torchvision.ops import box_iou as IoU
from torchvision.ops import complete_box_iou as CIoU
from torchvision.ops import distance_box_iou as DIoU
from torchvision.ops import generalized_box_iou as GIoU
from torchvision.ops.boxes import _upcast

Item_Bbox = Literal["ltrb", "ltwh", "xywh"]

T = TypeVar("T")


def flat_box_inter_union(ltrb_a: torch.Tensor, ltrb_b: torch.Tensor) -> torch.Tensor:
    assert len(ltrb_a) == len(ltrb_b), "输入的两个张量必须一致"
    area1 = box_area(ltrb_a)
    area2 = box_area(ltrb_b)
    lt = torch.max(ltrb_a[:, :2], ltrb_b[:, :2])  # [N, 2]
    rb = torch.min(ltrb_a[:, 2:], ltrb_b[:, 2:])  # [N, 2]
    wh = _upcast(rb - lt).clamp(min=0)  # [N, 2]
    inter = wh[:, 0] * wh[:, 1]  # [N]
    union = area1 + area2 - inter
    return inter, union


def flat_IoU(ltrb_a: torch.Tensor, ltrb_b: torch.Tensor) -> torch.Tensor:
    """
    计算两个框的 IoU（Intersection over Union）
    Args:
        ltrb_a: 第一个框的坐标 [L, T, R, B], shape=(N, 4)
        ltrb_b: 第二个框的坐标 [L, T, R, B], shape=(N, 4)
    Returns:
        IoU: 两个框的 IoU，shape=(N,)
    """

    # 计算交集的左上角和右下角坐标
    inter, union = flat_box_inter_union(ltrb_a, ltrb_b)
    iou = inter / union

    return iou


def flat_GIoU(ltrb_a: torch.Tensor, ltrb_b: torch.Tensor) -> torch.Tensor:
    """
    计算两个框的 GIoU（Generalized Intersection over Union）
    Args:
        ltrb_a: 第一个框的坐标 [L, T, R, B], shape=(N, 4)
        ltrb_b: 第二个框的坐标 [L, T, R, B], shape=(N, 4)
    Returns:
        GIoU: 两个框的 GIoU，shape=(N,)
    """

    inter, union = flat_box_inter_union(ltrb_a, ltrb_b)
    iou = inter / union
    lti = torch.min(ltrb_a[:, :2], ltrb_b[:, :2])
    rbi = torch.max(ltrb_a[:, 2:], ltrb_b[:, 2:])
    whi = _upcast(rbi - lti).clamp(min=0)  # [N, 2]
    areai = whi[:, 0] * whi[:, 1]

    return iou - (areai - union) / areai


def flat_CIoU(ltrb_a: torch.Tensor, ltrb_b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    计算两个框的 CIoU（Complete Intersection over Union）
    Args:
        ltrb_a: 第一个框的坐标 [L, T, R, B], shape=(N, 4)
        ltrb_b: 第二个框的坐标 [L, T, R, B], shape=(N, 4)
    Returns:
        CIoU: 两个框的 CIoU，shape=(N,)
    """
    ltrb_a = _upcast(ltrb_a)
    ltrb_b = _upcast(ltrb_b)
    diou, iou = flat_box_diou_iou(ltrb_a, ltrb_b, eps)

    w_pred = ltrb_a[:, 2] - ltrb_a[:, 0]
    h_pred = ltrb_a[:, 3] - ltrb_a[:, 1]

    w_gt = ltrb_b[:, 2] - ltrb_b[:, 0]
    h_gt = ltrb_b[:, 3] - ltrb_b[:, 1]

    v = (4 / (torch.pi**2)) * torch.pow(torch.atan(w_pred / h_pred) - torch.atan(w_gt / h_gt), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return diou - alpha * v


def flat_DIoU(ltrb_a: torch.Tensor, ltrb_b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    计算两个框的 DIoU（Distance Intersection over Union）
    Args:
        ltrb_a: 第一个框的坐标 [L, T, R, B], shape=(N, 4)
        ltrb_b: 第二个框的坐标 [L, T, R, B], shape=(N, 4)
    Returns:
        DIoU: 两个框的 DIoU，shape=(N,)
    """
    ltrb_a = _upcast(ltrb_a)
    ltrb_b = _upcast(ltrb_b)
    diou, _ = flat_box_diou_iou(ltrb_a, ltrb_b, eps=eps)
    return diou


def flat_box_diou_iou(ltrb_a: torch.Tensor, ltrb_b: torch.Tensor, eps: float = 1e-7) -> tuple[torch.Tensor, torch.Tensor]:

    iou = flat_IoU(ltrb_a, ltrb_b)
    lti = torch.min(ltrb_a[:, :2], ltrb_b[:, :2])
    rbi = torch.max(ltrb_a[:, 2:], ltrb_b[:, 2:])
    whi = _upcast(rbi - lti).clamp(min=0)  # [N, 2]
    diagonal_distance_squared = (whi[:, 0] ** 2) + (whi[:, 1] ** 2) + eps
    # centers of boxes
    x_p = (ltrb_a[:, 0] + ltrb_a[:, 2]) / 2
    y_p = (ltrb_a[:, 1] + ltrb_a[:, 3]) / 2
    x_g = (ltrb_b[:, 0] + ltrb_b[:, 2]) / 2
    y_g = (ltrb_b[:, 1] + ltrb_b[:, 3]) / 2
    # The distance between boxes' centers squared.
    centers_distance_squared = (_upcast((x_p - x_g)) ** 2) + (_upcast((y_p - y_g)) ** 2)
    # The distance IoU is the IoU penalized by a normalized
    # distance between boxes' centers squared.
    return iou - (centers_distance_squared / diagonal_distance_squared), iou


"""

默认不加载Torch，以提高运行效率
但是支持转换 torch.Tensor 的数据

#? 多重循环小批量矩阵运算 速度测试， 循环条件 500*24*60，三层循环，矩阵大小为 40
# * 总结，inplace更快，速度上，
#! Numpy >> Tensor(GPU) > Tensor(CPU)
# * 耗时如下
['numpy    ']: 0.0131 s.
['numpy (inplace=True)']: 0.0110 s.
['tensor(CPU)         ']: 0.1550 s.
['(V)tensor(CPU)      ']: 0.0603 s.
['tensor(GPU)         ']: 0.2785 s.
['(V)tensor(GPU)      ']: 0.1147 s.
['tensor(CPU) (inplace=True)']: 0.1057 s.
['tensor(GPU) (inplace=True)']: 0.1737 s.

#? 大批量矩阵运算 循环条件速度测试， 循环条件 46*24，两层循环，矩阵大小为 1000000

# * 总结，inplace更快，速度上，
#! Tensor(GPU) > Tensor(CPU) >> Numpy
# * 耗时如下
['numpy    ']: 9.3960 s.
['numpy (inplace=True)']: 6.8870 s.
['tensor(CPU)         ']: 1.0695 s.
['(V)tensor(CPU)      ']: 0.9479 s.
['tensor(GPU)         ']: 0.2654 s.
['(V)tensor(GPU)      ']: 0.1167 s.
['tensor(CPU) (inplace=True)']: 0.7182 s.
['tensor(GPU) (inplace=True)']: 0.1646 s.

"""


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


def pt_ltrb_to_ltwh(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (l, t, r, b) format to (l, t, w, h) format.
    (l, t) refers to top left of bounding box.
    (r, b) refer to bottom right of bounding box
    Args:
        boxes (Tensor[N, 4]): boxes in (l, t, r, b) which will be converted.

    Returns:
        boxes (Tensor[N, 4]): boxes in (l, t, w, h) format.
    """
    l, t, r, b = boxes.unbind(-1)
    w = r - l  # x2 - x1
    h = b - t  # y2 - y1
    boxes = torch.stack((l, t, w, h), dim=-1)
    return boxes


def ltwh_to_ltrb(ltwh: T, inplace=False) -> T:
    ndim1, ltrb = __quick_convert(ltwh, inplace)

    ltrb[:, 2] = ltrb[:, 2] + ltrb[:, 0]
    ltrb[:, 3] = ltrb[:, 3] + ltrb[:, 1]
    if ndim1:
        return ltrb[0]
    return ltrb


def pt_ltwh_to_ltrb(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (l, t, w, h) format to (l, t, r, b) format.
    (l, t) refers to top left of bounding box.
    (w, h) refers to width and height of box.
    Args:
        boxes (Tensor[N, 4]): boxes in (x, y, w, h) which will be converted.

    Returns:
        boxes (Tensor[N, 4]): boxes in (l, t, r, b) format.
    """
    l, t, w, h = boxes.unbind(-1)
    boxes = torch.stack([l, t, l + w, t + h], dim=-1)
    return boxes


def ltwh_to_xywh(ltwh: T, inplace=False) -> T:
    """
    左上宽高 转 中心宽高
    """
    ndim1, xywh = __quick_convert(ltwh, inplace)

    xywh[:, 0] = xywh[:, 0] + 0.5 * xywh[:, 2]
    xywh[:, 1] = xywh[:, 1] + 0.5 * xywh[:, 3]
    if ndim1:
        return xywh[0]
    return xywh


def pt_ltwh_to_xywh(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (l, t, w, h) format to (cx, cy, w, h) format.
    (l, t) refers to top left of bounding box.
    (w, h) refers to width and height of box.
    Args:
        boxes (Tensor(N, 4)): boxes in (l, t, w, h) format which will be converted.

    Returns:
        boxes (Tensor[N, 4]): boxes in (cx, cy, w, h) format.
    """
    # We need to change all 4 of them so some temporary variable is needed.
    l, t, w, h = boxes.unbind(-1)
    cx = l + 0.5 * w
    cy = t + 0.5 * h

    return torch.stack((cx, cy, w, h), dim=-1)


def xywh_to_ltwh(xywh: T, inplace=False) -> T:
    """ """
    ndim1, ltwh = __quick_convert(xywh, inplace)

    ltwh[:, 0] = ltwh[:, 0] - 0.5 * ltwh[:, 2]  # L
    ltwh[:, 1] = ltwh[:, 1] - 0.5 * ltwh[:, 3]  # T
    if ndim1:
        return ltwh[0]
    return ltwh


def pt_xywh_to_ltwh(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (cx, cy, w, h) format to (l, t, w, h) format.
    (cx, cy) refers to center of bounding box
    (w, h) are width and height of bounding box
    Args:
        boxes (Tensor[N, 4]): boxes in (cx, cy, w, h) format which will be converted.

    Returns:
        boxes (Tensor(N, 4)): boxes in (l, t, w, h) format.
    """
    # We need to change all 4 of them so some temporary variable is needed.
    cx, cy, w, h = boxes.unbind(-1)
    l = cx - 0.5 * w
    t = cy - 0.5 * h

    return torch.stack((l, t, w, h), dim=-1)


def xywh_to_ltrb(xywh: T, inplace=False) -> T:
    """ """
    ndim1, ltrb = __quick_convert(xywh, inplace)

    ltrb[:, 0] = ltrb[:, 0] - 0.5 * ltrb[:, 2]  # L
    ltrb[:, 1] = ltrb[:, 1] - 0.5 * ltrb[:, 3]  # T
    ltrb[:, 2] = ltrb[:, 2] + ltrb[:, 0]
    ltrb[:, 3] = ltrb[:, 3] + ltrb[:, 1]
    if ndim1:
        return ltrb[0]
    return ltrb


def pt_xywh_to_ltrb(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (cx, cy, w, h) format to (l, t, r, b) format.
    (x, y) refers to center of bounding box
    (w, h) are width and height of bounding box
    Args:
        boxes (Tensor[N, 4]): boxes in (cx, cy, w, h) format which will be converted.

    Returns:
        boxes (Tensor(N, 4)): boxes in (l, t, r, b) format.
    """
    # We need to change all 4 of them so some temporary variable is needed.
    cx, cy, w, h = boxes.unbind(-1)
    l = cx - 0.5 * w
    t = cy - 0.5 * h
    r = cx + 0.5 * w
    b = cy + 0.5 * h

    boxes = torch.stack((l, t, r, b), dim=-1)

    return boxes


def ltrb_to_xywh(ltrb: T, inplace=False) -> T:
    ndim1, xywh = __quick_convert(ltrb, inplace)

    xywh[:, 2] = xywh[:, 2] - xywh[:, 0]  # W = R - L  # LTWB
    xywh[:, 3] = xywh[:, 3] - xywh[:, 1]  # H = B - T  # LTWH
    xywh[:, 0] = xywh[:, 0] + 0.5 * xywh[:, 2]  # X = L + W / 2 # XTWH
    xywh[:, 1] = xywh[:, 1] + 0.5 * xywh[:, 3]  # Y = B + H / 2 # XYWH
    if ndim1:
        return xywh[0]
    return xywh


def pt_ltrb_to_xywh(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (l, t, r, b) format to (cx, cy, w, h) format.
    (l, t) refers to top left of bounding box.
    (r, b) refer to bottom right of bounding box
    Args:
        boxes (Tensor[N, 4]): boxes in (l, t, r, b) format which will be converted.

    Returns:
        boxes (Tensor(N, 4)): boxes in (cx, cy, w, h) format.
    """
    l, t, r, b = boxes.unbind(-1)
    cx = (l + r) / 2
    cy = (t + b) / 2
    w = r - l
    h = b - t

    boxes = torch.stack((cx, cy, w, h), dim=-1)

    return boxes


dict_convert_fn = {
    "ltrb": {
        "ltwh": ltrb_to_ltwh,
        "xywh": ltrb_to_xywh,
        "ltrb": lambda x, inplace=False: x if inplace else deepcopy(x),
    },
    "ltwh": {
        "ltrb": ltwh_to_ltrb,
        "xywh": ltwh_to_xywh,
        "ltwh": lambda x, inplace=False: x if inplace else deepcopy(x),
    },
    "xywh": {
        "ltrb": xywh_to_ltrb,
        "ltwh": xywh_to_ltwh,
        "xywh": lambda x, inplace=False: x if inplace else deepcopy(x),
    },
    ("ltrb", "ltwh"): ltrb_to_ltwh,
    ("ltrb", "xywh"): ltrb_to_xywh,
    ("ltrb", "ltrb"): lambda x, inplace=False: x if inplace else deepcopy(x),
    ("ltwh", "ltrb"): ltwh_to_ltrb,
    ("ltwh", "xywh"): ltwh_to_xywh,
    ("ltwh", "ltwh"): lambda x, inplace=False: x if inplace else deepcopy(x),
    ("xywh", "ltrb"): xywh_to_ltrb,
    ("xywh", "ltwh"): xywh_to_ltwh,
    ("xywh", "xywh"): lambda x, inplace=False: x if inplace else deepcopy(x),
}


if __name__ == "__main__":
    import torch
    from pretty_tools.echo import X_Timer
    from torchvision.ops import box_convert

    def wrap_fn(data, N, bs, inplace=False):
        for _ in range(N):
            for _ in range(bs):
                xywh_to_ltrb(data, inplace)

    def wrap_fn_torchvision(data, N, bs):
        for _ in range(N):
            for _ in range(bs):
                pt_xywh_to_ltrb(data)

    def test_speed_multi_for_small_array_convert():
        """
        #? 小批量多重循环条件速度测试， 循环条件 46*24*60，三层循环，矩阵大小为 40

        # * 总结，inplace更快，速度上，Numpy > Tensor(GPU) > Tensor(CPU)
        # * 耗时如下
        ['numpy (inplace=True)']:        5.3548 s
        ['numpy ']:                      6.5552 s
        ['tensor(CPU) (inplace=True)']:  51.7493 s
        ['tensor(CPU)']:                 77.1919 s
        ['tensor(GPU) (inplace=True)']:  87.6123 s
        ['tensor(GPU)']:                 125.9889 s

        """

        N = 50
        bs = 24
        repeat = 60

        n_bbox = 40
        # 生成
        random_bbox_np = np.random.rand(n_bbox, 4).astype(np.float32)
        random_bbox_tensor = torch.randn(n_bbox, 4, dtype=torch.float32)
        random_bbox_tensor_gpu = random_bbox_tensor.to("cuda")

        timer = X_Timer()

        wrap_fn(random_bbox_np, N, bs)
        timer.record("numpy", verbose=True)

        wrap_fn(random_bbox_np, N, bs, inplace=True)
        timer.record("numpy (inplace=True)", verbose=True)

        wrap_fn(random_bbox_tensor, N, bs)
        timer.record("tensor(CPU)", verbose=True)
        wrap_fn_torchvision(random_bbox_tensor, N, bs)
        timer.record("(V)tensor(CPU)", verbose=True)

        wrap_fn(random_bbox_tensor_gpu, N, bs)
        timer.record("tensor(GPU)", verbose=True)
        wrap_fn_torchvision(random_bbox_tensor_gpu, N, bs)
        timer.record("(V)tensor(GPU)", verbose=True)

        wrap_fn(random_bbox_tensor, N, bs, inplace=True)
        timer.record("tensor(CPU) (inplace=True)", verbose=True)

        wrap_fn(random_bbox_tensor_gpu, N, bs, inplace=True)
        timer.record("tensor(GPU) (inplace=True)", verbose=True)

    def test_speed_huge_array_convert():
        """
        #? 大批量矩阵运算 循环条件速度测试， 循环条件 46*24，两层循环，矩阵大小为 1000000

        # * 总结，inplace更快，速度上，Tensor(GPU) > Tensor(CPU) > Numpy
        # * 耗时如下
        ['tensor(GPU) (inplace=True)']: 0.1369 s
        ['tensor(GPU)']:                0.2200 s
        ['tensor(CPU) (inplace=True)']: 0.5653 s
        ['tensor(CPU)']:                0.9554 s.
        ['numpy (inplace=True)']:       7.9395 s
        ['numpy']:                      11.2856 s

        """

        N = 46
        bs = 24

        n_bbox = 1000000
        # 生成
        random_bbox_np = np.random.rand(n_bbox, 4).astype(np.float32)
        random_bbox_tensor = torch.randn(n_bbox, 4, dtype=torch.float32)
        random_bbox_tensor_gpu = random_bbox_tensor.to("cuda")

        timer = X_Timer()

        wrap_fn(random_bbox_np, N, bs)
        timer.record("numpy", verbose=True)

        wrap_fn(random_bbox_np, N, bs, inplace=True)
        timer.record("numpy (inplace=True)", verbose=True)

        wrap_fn(random_bbox_tensor, N, bs)
        timer.record("tensor(CPU)", verbose=True)
        wrap_fn_torchvision(random_bbox_tensor, N, bs)
        timer.record("(V)tensor(CPU)", verbose=True)

        wrap_fn(random_bbox_tensor_gpu, N, bs)
        timer.record("tensor(GPU)", verbose=True)
        wrap_fn_torchvision(random_bbox_tensor_gpu, N, bs)
        timer.record("(V)tensor(GPU)", verbose=True)

        wrap_fn(random_bbox_tensor, N, bs, inplace=True)
        timer.record("tensor(CPU) (inplace=True)", verbose=True)

        wrap_fn(random_bbox_tensor_gpu, N, bs, inplace=True)
        timer.record("tensor(GPU) (inplace=True)", verbose=True)

    print("========== test_speed_multi_for_small_array_convert =======")
    test_speed_multi_for_small_array_convert()
    print("========== test_speed_huge_array_convert =======")
    test_speed_huge_array_convert()
