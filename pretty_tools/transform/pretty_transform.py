from typing import Any, Callable, List, Tuple

import imgaug as ia
from imgaug import augmenters as iaa

"""
这里是用CPU做数据预处理的方法

这里的工具是能够同时变换标签以及图像，麻烦的地方在于格式处理

* OpenCV 格式的 np.ndarray

* Pillow 格式的 PIL.Image

* Pytorch 格式的 torch.Tensor

"""
import random
from typing import Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image

from ..datastruct import GeneralAnn, TrackCameraInstances


class Base_Transform:
    def __repr__(self):
        return f"{self.__class__.__name__}()"


class PIL_To_CV2(Base_Transform):
    def __init__(self):
        pass

    def __call__(
        self, image: Image.Image, bounding_boxes=None
    ) -> Tuple[np.ndarray, Any]:
        """
        这里不涉及颜色转换
        """
        image = np.asarray(image)  # type: ignore

        return image, bounding_boxes  # type: ignore


class Img_To_Tensor(Base_Transform):
    """
    将图像转换为Tensor，不转换标签

    可转换 PIL.Image, np.ndarray, torch.Tensor

    注意: np.ndarray 默认是RGB格式
    """

    def __init__(self) -> None:
        self.func = torchvision.transforms.ToTensor()
        self.pil_func = torchvision.transforms.PILToTensor()

    def __call__(
        self, image: Union[Image.Image, np.ndarray, torch.Tensor], bounding_boxes=None
    ) -> Tuple[torch.Tensor, Any]:
        if isinstance(image, torch.Tensor):
            return image, bounding_boxes
        elif isinstance(image, np.ndarray):
            image = self.func(image)
        elif isinstance(image, Image.Image):
            image = self.func(image)
        else:
            return TypeError()
        return image, bounding_boxes


class Img_To_Numpy(Base_Transform):
    def __call__(
        self, image: Union[Image.Image, np.ndarray, torch.Tensor], bounding_boxes=None
    ) -> Tuple[torch.Tensor, Any]:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        elif isinstance(image, np.ndarray):
            return image, bounding_boxes
        elif isinstance(image, Image.Image):
            image = np.asarray(image)
        else:
            return TypeError()
        return image, bounding_boxes

    pass


#! =============================================   标注实例转换  =============================================
class Pandas_To_Instances(Base_Transform):
    def __call__(self, image, bounding_boxes=None) -> Tuple[Any, Any]:
        if bounding_boxes is not None:
            assert isinstance(bounding_boxes, pd.DataFrame)
            bounding_boxes = TrackCameraInstances(bounding_boxes, str_format="xywh")
        return image, bounding_boxes


class Numpy_To_BoundingBoxes(Base_Transform):
    def __call__(self, image, bounding_boxes=None) -> Tuple[Any, Any]:
        if bounding_boxes is not None:
            if isinstance(image, np.ndarray):
                shape = image.shape  # hw
            if isinstance(image, Image.Image):
                shape = image.size[1], image.size[0]  # hw
            bounding_boxes = ia.BoundingBoxesOnImage.from_xyxy_array(
                bounding_boxes, shape=shape
            )
        return image, bounding_boxes


class Numpy_To_Instance(Base_Transform):
    def __call__(self, image, bounding_boxes=None) -> Tuple[Any, Any]:
        return NotImplementedError()
        if bounding_boxes is not None:
            pass
        return image, bounding_boxes


class Instances_To_BoundingBoxes(Base_Transform):
    def __call__(self, image, bounding_boxes=None) -> Tuple[Any, Any]:
        if bounding_boxes is not None:
            if isinstance(image, np.ndarray):
                shape = image.shape  # hw
            if isinstance(image, Image.Image):
                shape = image.size[1], image.size[0]  # hw
            assert isinstance(bounding_boxes, GeneralAnn)
            bounding_boxes = ia.BoundingBoxesOnImage.from_xyxy_array(
                bounding_boxes.ltrb, shape=shape
            )

        return image, bounding_boxes


class BoundingBoxes_To_Numpy(Base_Transform):
    def __call__(self, image, bounding_boxes=None):
        if bounding_boxes is not None:
            bounding_boxes = ia.BoundingBoxesOnImage.to_xyxy_array(bounding_boxes)
        return image, bounding_boxes


# =============================================
class Resize(Base_Transform):
    """
    必须是cv2格式(np.ndarray)才能进行转换，因为使用的是imgaug库

    >>> __call__(image: np.ndarray, bounding_boxes=None):

    """

    def __init__(
        self, size: "int | List[int]", max_size=None, interpolation="cubic"
    ) -> None:
        if isinstance(size, int):
            self.size = [size]
        else:
            self.size = size
        self.interpolation = interpolation
        self.max_size = max_size

        self.resize_fix = iaa.Resize(
            {"shorter-side": size, "longer-side": "keep-aspect-ratio"}
        )
        if self.max_size is not None:
            self.resize_fix_max = iaa.Resize(
                {"shorter-side": "keep-aspect-ratio", "longer-side": self.max_size}
            )

    def __call__(self, image: np.ndarray, bounding_boxes=None) -> Tuple[Any, Any]:
        image, bounding_boxes = self.resize_fix(image=image, bounding_boxes=bounding_boxes)  # type: ignore
        if self.max_size is not None:
            if max(image.shape[:2]) > self.max_size:
                # todo 这里有点麻烦了，如果变换了一次后图像的大小超过了max_size，那么就需要再变换一次
                image, bounding_boxes = self.resize_fix_max(image=image, bounding_boxes=bounding_boxes)  # type: ignore
        return image, bounding_boxes

    def __repr__(self) -> str:
        detail = f"size={self.size}, max_size={self.max_size}, interpolation={self.interpolation}"
        return f"{self.__class__.__name__}({detail})"


# * ======================    Pad   Start    ===============================
class Pad(Base_Transform):
    list_pad_mode = ["constant", "edge", "reflect", "symmetric"]
    list_position = [
        "uniform",
        "normal",
        "center",
        "left-top",
        "left-center",
        "left-bottom",
        "center-top",
        "center-center",
        "center-bottom",
        "right-top",
        "right-center",
        "right-bottom",
    ]

    def __init__(self) -> None:
        pass


class PadToFixedSize(Pad):
    def __init__(
        self,
        width: int,
        height: int,
        pad_mode="constant",
        pad_cval=0,
        position="center",
    ) -> None:
        self.width = width
        self.height = height
        self.aug = iaa.PadToFixedSize(
            width=width,
            height=height,
            pad_mode=pad_mode,
            pad_cval=pad_cval,
            position=position,
        )
        assert pad_mode in self.list_pad_mode

    def __call__(
        self, image: np.ndarray, bounding_boxes=None
    ) -> Tuple[np.ndarray, Any]:
        assert type(image) == np.ndarray
        image, bounding_boxes = self.aug(image=image, bounding_boxes=bounding_boxes)  # type: ignore
        return image, bounding_boxes


# * ======================    Pad    END    ===============================


class RGB_Offset(Base_Transform):
    """ """

    def __init__(self, rgb_offset) -> None:
        super().__init__()
        self.rgb_offset = rgb_offset

    def __call__(
        self, image: "Image.Image | np.ndarray | torch.Tensor", bounding_boxes=None
    ) -> Tuple[Any, Any]:
        if isinstance(image, np.ndarray):
            image += np.array(self.rgb_offset)
        if isinstance(image, torch.Tensor):
            image += torch.tensor(self.rgb_offset)[:, None, None]
        # todo 还没补全其他类型的处理方式
        return image, bounding_boxes

    def __repr__(self) -> str:
        detail = f"rgb_offset={self.rgb_offset}"
        return f"{self.__class__.__name__}({detail})"


class RGB_Rescale(Base_Transform):
    """
    除以数值的缩放倍率
    """

    def __init__(self, scale_frac: "int | float | np.ndarray | List" = 1 / 255) -> None:
        if isinstance(scale_frac, List):
            self.scale_frac = np.array(scale_frac)
        else:
            self.scale_frac = np.array([scale_frac])

    def __call__(
        self, image: "Image.Image | np.ndarray | torch.Tensor", bounding_boxes=None
    ) -> Tuple[Any, Any]:
        if isinstance(image, np.ndarray):
            image /= self.scale_frac

        if isinstance(image, torch.Tensor):
            image /= torch.tensor(self.scale_frac)[:, None, None]

        # todo 还没补全其他类型的处理方式
        return image, bounding_boxes

    def __repr__(self) -> str:
        detail = f"scale_frac={self.scale_frac}"
        return f"{self.__class__.__name__}({detail})"


class RGB_Scale(Base_Transform):
    """
    乘以数值的缩放倍率
    """

    def __init__(self, scale_frac: "int | float | np.ndarray | List" = 1 / 255) -> None:
        if isinstance(scale_frac, List):
            self.scale_frac = np.array(scale_frac)
        else:
            self.scale_frac = np.array([scale_frac])

    def __call__(
        self, image: "Image.Image | np.ndarray | torch.Tensor", bounding_boxes=None
    ) -> Tuple[Any, Any]:
        if isinstance(image, np.ndarray):
            image *= self.scale_frac
        if isinstance(image, torch.Tensor):
            image *= torch.tensor(self.scale_frac)[:, None, None]

        # todo 还没补全其他类型的处理方式
        return image, bounding_boxes

    def __repr__(self) -> str:
        detail = f"scale_frac={self.scale_frac}"
        return f"{self.__class__.__name__}({detail})"


class Normalize(Base_Transform):
    """
    正则归一化，输出格式一般为torch
    """

    def __init__(self, mean, std, inplace=False) -> None:
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(
        self, image: "Image.Image | np.ndarray | torch.Tensor", bounding_boxes=None
    ) -> Tuple[torch.Tensor, Any]:
        if isinstance(image, torch.Tensor):
            image = F.normalize(image, self.mean, self.std, self.inplace)
        elif isinstance(image, np.ndarray):
            image = torchvision.transforms.ToTensor()(image)
            image = F.normalize(image, self.mean, self.std, self.inplace)
        else:
            raise NotImplementedError(f"未补全类型{type(image)}的归一化方法")
        return image, bounding_boxes

    def __repr__(self) -> str:
        detail = f"mean={self.mean}, std={self.std}, inplace={self.inplace}"
        return f"{self.__class__.__name__}({detail})"


class Compose:
    def __init__(self, list_transforms_func: "List[Callable] | None" = None):
        if list_transforms_func is None:
            list_transforms_func = []
        self.list_transforms_func = list_transforms_func

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.list_transforms_func:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

    def append(self, transforms_func: Callable):
        self.list_transforms_func.append(transforms_func)
        return self

    def append_left(self, transforms_func: Callable):
        self.list_transforms_func.insert(0, transforms_func)
        return self

    def join(self, list_transforms_func: List[Callable]):
        self.list_transforms_func += list_transforms_func
        return self

    def __call__(self, imgs, targets=None) -> Tuple[torch.Tensor, Any]:
        for t in self.list_transforms_func:
            imgs, targets = t(image=imgs, bounding_boxes=targets)
        return imgs, targets
