import copy
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ...resources import path_font_arial
from ..core import log_visualization
from .draw_base import draw_bboxes, draw_gt_bboxes, draw_title


class Visual_Track:
    """

    输入的可以是Image格式，也可以是np.ndarray格式，如果输入的是np.ndarray格式，则默认是BGR格式

    """

    def __init__(self, outline=1, mask: float = 0.5) -> None:
        """
        num_classes 如果是0，则默认没有跟踪的类，所有目标都根据id分配一个颜色， 如果是1，则所有类有唯一对应的颜色
        bbox 必须是一个字典，有词条 id score class
        """
        pass
        self.dict_color = {}
        self.outline = outline
        self.mask = mask

        self.font_size = 10

    @staticmethod
    def convert_img(img):
        if type(img) == np.ndarray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        assert type(img) == Image.Image
        return img

    @staticmethod
    def convert_ori_box(ori_size: Tuple[int, int], bboxes):
        height, width = ori_size
        bboxes = copy.deepcopy(bboxes)
        # 判断是否是归一化的，如果所有数据都小于1，且是浮点数，且不都为零，则认为是归一化的
        if np.all(bboxes > -1) > 0 and np.all(bboxes < 2):  # * 这里并没要要求普遍小于1，因为可能存在着有坐标稍稍超过一些，所以限定到2和-1
            bboxes[:, [1, 3]] *= height
            bboxes[:, [0, 2]] *= width
        return bboxes

    def draw_check(self, img, bboxes, classes, outline):
        img = self.convert_img(img)

        outline = self.outline if outline is None else outline
        if len(bboxes) != 0:
            bboxes = self.convert_ori_box((img.height, img.width), bboxes)

        return img, bboxes, classes, outline

    def draw_tracks(
        self,
        img: "Image.Image | np.ndarray",
        bboxes: np.ndarray,
        ids: np.ndarray = None,  # type: ignore
        classes: np.ndarray = None,  # type: ignore
        probs: np.ndarray = None,  # type: ignore
        outline=None,
        mask=False,
    ) -> Image.Image:
        """

        Args:
            img (Image.Image | np.ndarray): BGR 格式的图像, 或者是Image格式的图像，如果是np.ndarray格式，则会自动转换为RGB格式
            bboxes (np.ndarray): 格式为 LTRB
            ids (np.ndarray, optional): 目标ID数组. Defaults to None.
            classes (_type_, optional): 目标类别数组. Defaults to None.
            probs (_type_, optional): 目标置信度数组. Defaults to None.
            outline (_type_, optional): 边框大小，在实例化的时候可以给定一个默认值. Defaults to None.
            mask (bool, optional): 是否在边框内部添加一定的蒙版. Defaults to False.

        Returns:
            Image.Image: _description_
        """
        img, bboxes, classes, outline = self.draw_check(img, bboxes, classes, outline)
        if len(bboxes) == 0:
            log_visualization.warning("无标注信息，忽略可视化标注")
            return img
        if ids is not None:
            assert len(bboxes) == len(ids)

        #! 生成info
        infos = ["" for _ in bboxes]

        if ids is not None:
            for i, _id in enumerate(ids):
                infos[i] += f"@id: {_id} "
        else:
            ids = [0 for _ in bboxes]  # type: ignore
        if classes is not None:
            assert len(bboxes) == len(classes)
            for i, cls in enumerate(classes):
                infos[i] += f"@cls: {cls} "

        if probs is not None:
            assert len(bboxes) == len(probs)
            for i, prob in enumerate(probs):
                infos[i] += f"@prob: {prob:.2f} "
        if mask is True:
            mask = self.mask
        elif type(mask) in [float, int]:
            mask = mask

        img = draw_bboxes(img, bboxes, self.get_color(ids), infos=infos, outline=outline, mask=mask)
        return img

    def draw_dets(self, img: "Image.Image | np.ndarray", bboxes, classes=None, probs=None, color=None, outline=None) -> Image.Image:
        img, bboxes, classes, outline = self.draw_check(img, bboxes, classes, outline)
        if len(bboxes) == 0:
            log_visualization.warning("无标注信息，忽略可视化标注")
            return img
        #! 生成info
        infos = ["" for _ in bboxes]
        if classes is not None:
            assert len(bboxes) == len(classes)
            for i, cls in enumerate(classes):
                infos[i] += f"@cls: {cls} "
        if probs is not None:
            assert len(bboxes) == len(probs)
            for i, prob in enumerate(probs):
                infos[i] += f"@prob: {prob:.2f} "
        img = draw_bboxes(img, bboxes, color, infos=infos, outline=outline)
        return img

    def draw_gt(self, img: "Image.Image | np.ndarray", bboxes, classes=None, outline=None) -> Image.Image:
        """
        绘制gt
        """
        img, bboxes, classes, outline = self.draw_check(img, bboxes, classes, outline)
        if len(bboxes) == 0:
            log_visualization.warning("无标注信息，忽略可视化标注")
            return img
        #! 生成info
        infos = [f"[gt]@ " for _ in bboxes]
        img = draw_gt_bboxes(img, bboxes, infos, outline=outline)
        return img

    def draw_info(self, img: "Image.Image | np.ndarray", info: str, size_font: int = 24):
        """
        绘制整个图像的信息
        Args:
            img (Image.Image | np.ndarray): _description_
            title (str): 信息字符串，用换行符换行，会自动切割
            size_font (int, optional): _description_. Defaults to 24.

        Returns:
            _type_: _description_
        """
        img = draw_title(img, info, size_font / 24)
        return img

    def cut_box(self, img: "Image.Image | np.ndarray", bboxes):
        img = self.convert_img(img)
        bboxes = copy.deepcopy(bboxes)
        height = img.height
        width = img.width
        bboxes = self.convert_ori_box((height, width), bboxes)
        list_cut = []
        for bbox in bboxes:
            list_cut.append(img.crop(bbox))
        return list_cut

    def get_color(self, _ids):
        list_color = []
        for i in _ids:
            if i not in self.dict_color:
                self.dict_color[i] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            list_color.append(self.dict_color[i])
        return list_color

    def __repr__(self) -> str:
        return f"Visual_Track; num_color:{len(self.dict_color)}"

    def __call__(self, img: "Image.Image | np.ndarray", bboxes, _ids=None, classes=None, outline=None):
        return self.draw_tracks(img, bboxes, _ids, classes, outline)
