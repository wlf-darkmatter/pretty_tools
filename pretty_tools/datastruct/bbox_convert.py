"""

默认不加载Torch，以提高运行效率
但是支持转换 torch.Tensor 的数据

#? 多重循环小批量矩阵运算 速度测试， 循环条件 500*24*60，三层循环，矩阵大小为 40
# * 总结，inplace更快，速度上，
#! Numpy >> Tensor(GPU) > Tensor(CPU)
# * 耗时如下
['numpy (inplace=True)']:        0.5867 s   13.5x
['numpy ']:                      0.8959 s   8.88x
['tensor(CPU) (inplace=True)']:  4.7858 s   1.66x
['    tensor(CPU)     ']:        7.9595 s   1.00x
['tensor(GPU) (inplace=True)']:  8.2551 s   0.96x
['    tensor(GPU)     ']:        11.8812 s  0.67x

#? 大批量矩阵运算 循环条件速度测试， 循环条件 46*24，两层循环，矩阵大小为 1000000

# * 总结，inplace更快，速度上，
#! Tensor(GPU) > Tensor(CPU) >> Numpy
# * 耗时如下
['tensor(GPU) (inplace=True)']: 0.1369 s
['tensor(GPU)']:                0.2200 s
['tensor(CPU) (inplace=True)']: 0.5653 s
['tensor(CPU)']:                0.9554 s.
['numpy (inplace=True)']:       7.9395 s
['numpy']:                      11.2856 s


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
        from pretty_tools.echo import X_Timer
        from pretty_tools.datastruct.bbox_convert import ltrb_to_ltwh, ltrb_to_xywh, ltwh_to_ltrb, ltwh_to_xywh, xywh_to_ltrb, xywh_to_ltwh

        N = 500
        bs = 24
        repeat = 60

        n_bbox = 40
        # 生成
        random_bbox_np = np.random.rand(n_bbox, 4).astype(np.float32)
        random_bbox_tensor = torch.randn(n_bbox, 4, dtype=torch.float32)
        random_bbox_tensor_gpu = random_bbox_tensor.to("cuda")

        def wrap_fn(data, inplace=False):
            for _ in range(N):
                for _ in range(bs):
                    for _ in range(repeat):
                        xywh_to_ltrb(data, inplace)

        timer = X_Timer()

        wrap_fn(random_bbox_np)
        timer.record("numpy", verbose=True)

        wrap_fn(random_bbox_np, inplace=True)
        timer.record("numpy (inplace=True)", verbose=True)

        wrap_fn(random_bbox_tensor)
        timer.record("tensor(CPU)", verbose=True)

        wrap_fn(random_bbox_tensor_gpu)
        timer.record("tensor(GPU)", verbose=True)

        wrap_fn(random_bbox_tensor, inplace=True)
        timer.record("tensor(CPU) (inplace=True)", verbose=True)

        wrap_fn(random_bbox_tensor_gpu, inplace=True)
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
        from pretty_tools.echo import X_Timer
        from pretty_tools.datastruct.bbox_convert import ltrb_to_ltwh, ltrb_to_xywh, ltwh_to_ltrb, ltwh_to_xywh, xywh_to_ltrb, xywh_to_ltwh

        N = 46
        bs = 24

        n_bbox = 1000000
        # 生成
        random_bbox_np = np.random.rand(n_bbox, 4).astype(np.float32)
        random_bbox_tensor = torch.randn(n_bbox, 4, dtype=torch.float32)
        random_bbox_tensor_gpu = random_bbox_tensor.to("cuda")

        def wrap_fn(data, inplace=False):
            for _ in range(N):
                for _ in range(bs):
                    xywh_to_ltrb(data, inplace)

        timer = X_Timer()

        wrap_fn(random_bbox_np)
        timer.record("numpy", verbose=True)

        wrap_fn(random_bbox_np, inplace=True)
        timer.record("numpy (inplace=True)", verbose=True)

        wrap_fn(random_bbox_tensor)
        timer.record("tensor(CPU)", verbose=True)

        wrap_fn(random_bbox_tensor_gpu)
        timer.record("tensor(GPU)", verbose=True)

        wrap_fn(random_bbox_tensor, inplace=True)
        timer.record("tensor(CPU) (inplace=True)", verbose=True)

        wrap_fn(random_bbox_tensor_gpu, inplace=True)
        timer.record("tensor(GPU) (inplace=True)", verbose=True)
