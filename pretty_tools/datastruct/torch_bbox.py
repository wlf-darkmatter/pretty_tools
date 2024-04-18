import torch
from torchvision.ops import box_area
from torchvision.ops.boxes import _upcast
from torchvision.ops import generalized_box_iou as GIoU
from torchvision.ops import distance_box_iou as DIoU
from torchvision.ops import complete_box_iou as CIoU
from torchvision.ops import box_iou as IoU


def flat_box_inter_union(ltrb_a: torch.Tensor, ltrb_b: torch.Tensor) -> torch.Tensor:
    area1 = box_area(ltrb_a)
    area2 = box_area(ltrb_b)
    lt = torch.max(ltrb_a[:, :2], ltrb_b[:, :2])  # [N, 2]
    rb = torch.min(ltrb_a[:, 2:], ltrb_b[:, 2:])  # [N, 2]
    wh = _upcast(rb - lt).clamp(min=0)  # [N, 2]
    inter = wh[:, 0] * wh[:, 1]  # [N]
    union = area1 + area2 - inter
    return inter, union


def flat_iou(ltrb_a: torch.Tensor, ltrb_b: torch.Tensor) -> torch.Tensor:
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


def flat_giou(ltrb_a: torch.Tensor, ltrb_b: torch.Tensor) -> torch.Tensor:
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


def flat_ciou(ltrb_a: torch.Tensor, ltrb_b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
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


def flat_diou(ltrb_a: torch.Tensor, ltrb_b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
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

    iou = flat_iou(ltrb_a, ltrb_b)
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
