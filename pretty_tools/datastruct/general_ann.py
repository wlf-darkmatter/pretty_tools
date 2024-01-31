from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pandas import DataFrame
from PIL import Image
from pretty_tools.datastruct import bbox_convert_np as np_converter

track_column = ["frame", "id", "xc", "yc", "w", "h", "prob", "cls", "vis", "crowd"]
track_column_dtype = [int, int, np.float64, np.float64, np.float64, np.float64, np.float64, int, int, int]


def reformat_ann_dataframe(data_ann, str_format=None) -> DataFrame:
    """
    快速格式化一个二维数组
    todo 添加一个示例
    """
    if str_format is None:
        str_format = "xywh"
    if type(data_ann) == list:
        data_ann = np.array(data_ann)
    if isinstance(data_ann, DataFrame):
        data_ann = np.array(data_ann)
    if isinstance(data_ann, np.ndarray):
        if data_ann.shape[1] == 4:  # * 只有bbox，前面的信息用默认值补全
            data_ann = np.concatenate([np.zeros((data_ann.shape[0], 2)), data_ann], axis=1)
            data_ann[:, :2] = -1  # * 帧号和id 全部设置为 -1

    # * -----------------   为了方便，先把数据转换成 np.ndarray 格式，然后再转换成 dataframe 格式

    assert data_ann.shape[1] >= 6

    if data_ann.shape[1] < 7:
        data_ann = np.insert(data_ann, 6, 1.0, axis=1)  # prob
    if data_ann.shape[1] < 8:
        data_ann = np.insert(data_ann, 7, -1, axis=1)  # cls
    if data_ann.shape[1] < 9:
        data_ann = np.insert(data_ann, 8, -1, axis=1)  # vis
    if data_ann.shape[1] < 10:
        data_ann = np.insert(data_ann, 9, -1, axis=1)  # crowd

    data_ann = DataFrame({name: data_ann[:, i].astype(dtype) for i, (name, dtype) in enumerate(zip(track_column, track_column_dtype))})

    if str_format != "xywh":
        data_ann.loc[:, ["xc", "yc", "w", "h"]] = np_converter.dict_convert_fn[str_format]["xywh"](np.array(data_ann.iloc[:, 2:6]))

    return data_ann


class GeneralAnn:
    """
    #! 统一规定，不再设计针对单一一个锚框的变换类，没必要
    """

    # todo 针对这个类设计一个切片方法
    def __init__(
        self,
        ann: Optional[Union[np.ndarray, DataFrame, Sequence[np.ndarray], GeneralAnn]] = None,
        ori_ann: Optional[GeneralAnn] = None,
        str_format: Optional[str] = "xywh",
        ori_WH: Optional[Union[Tuple[int, int], np.ndarray]] = None,
        ori_img: Optional[Image.Image] = None,
        _move_copy=False,  #! 说明是否是移动构造
    ) -> None:
        """
        np.ndarray
        如果data的宽度为6，则认定其格式为 frame_id, target_id, *bbox
        如果data的宽度为4，则认定其格式为 *bbox

        ann: 归一化的数据
        ori_ann: 未归一化的原始数据
        """
        self.flag_norm = False
        self.__ori_WH = (1, 1)
        if ori_WH is not None:
            self.__ori_WH = tuple(ori_WH)

        self.ori_img = ori_img
        if ori_img is not None:
            if ori_WH is None:
                self.__ori_WH = ori_img.size
            else:
                if self.__ori_WH == ori_img.size:
                    pass
                else:
                    UserWarning(f"ori_WH={ori_WH} 与 ori_image.size={ori_img.size} 不一致")

        # * -------------------  构造 部分 --------------------------
        if not _move_copy:
            assert (ann is None) ^ bool(ori_ann is None), "GeneralAnn实例化时，ann 和 ori_ann 必须有且仅有一个为 None"
            if isinstance(ann, np.ndarray):
                assert ann.ndim == 2
            if ann is not None:
                self.flag_norm = True
                self.ann = reformat_ann_dataframe(ann, str_format=str_format)
            else:
                self.ann = reformat_ann_dataframe(ori_ann, str_format=str_format)

            if ori_WH is not None and ann is None:
                # * 同时输入了原始尺寸的锚框以及锚框对应的原图像的尺寸，这里就会根据需要进行自动归一化
                self.set_xywh(self.xywh / [*ori_WH, *ori_WH])  # type: ignore
                self.flag_norm = True  # * 标志已被归一化
        # * -------------------  移动构造 部分 --------------------------
        else:
            assert isinstance(ann, GeneralAnn), "移动构造必须保存传入的 ann 必须是 GeneralAnn 类型的 "
            assert ori_WH is None, "移动构造不允许传入 ori_WH, 应当在构造完毕之后设置，或者在构造基类的时候设置"
            assert str_format is None, "移动构造不允许传入 str_format, 应当在构造完毕之后设置，或者在构造基类的时候设置"
            assert ori_ann is None, "移动构造不允许传入 ori_ann, 应当在构造完毕之后设置，或者在构造基类的时候设置"
            assert ori_img is None, "移动构造不允许传入 ori_img, 应当在构造完毕之后设置，或者在构造基类的时候设置"
            self.ann = ann.ann  #! 确保调用的是setter或者直接修改对应的真值
            self.flag_norm = ann.flag_norm
            self.__ori_WH = ann.ori_WH
            self.ori_img = ann.ori_img

        # * ---------------------------------------------

    @property
    def shape(self):
        return self.ann.shape

    @property
    def ann(self):
        return self._ann

    @ann.setter
    def ann(self, value):
        self.get_frames.cache_clear()
        self.get_ids.cache_clear()
        self.get_ltrbs.cache_clear()
        self.get_xywhs.cache_clear()
        self.get_ltwhs.cache_clear()
        self.get_cls.cache_clear()
        self.get_probs.cache_clear()
        self._ann = reformat_ann_dataframe(value)

    # * ---------------------------------------------
    @property
    def ltrb(self):
        return self.get_ltrbs()

    def set_ltrb(self, value: np.ndarray):
        self.ann.iloc[:, 2:6] = np_converter.ltrb_to_xywh(value)
        self.get_ltrbs.cache_clear()

    @lru_cache(maxsize=1)
    def get_ltrbs(self):
        return np_converter.xywh_to_ltrb(self.ann.iloc[:, 2:6])

    # * ---------------------------------------------
    @property
    def xywh(self):
        return self.get_xywhs()

    def set_xywh(self, value: np.ndarray):
        self.ann.iloc[:, 2:6] = value
        self.get_xywhs.cache_clear()

    @lru_cache(maxsize=1)
    def get_xywhs(self):
        return np.array(self.ann.iloc[:, 2:6])

    # * ---------------------------------------------
    @property
    def ltwh(self):
        return self.get_ltwhs()

    def set_ltwh(self, value: np.ndarray):
        self.ann.iloc[:, 2:6] = np_converter.ltwh_to_xywh(value)
        self.get_ltwhs.cache_clear()

    @lru_cache(maxsize=1)
    def get_ltwhs(self):
        return np_converter.xywh_to_ltwh(self.ann.iloc[:, 2:6])

    # * ---------------------------------------------
    @property
    def frames(self):
        return self.get_frames()

    def set_frames(self, value: np.ndarray):
        self.ann.iloc[:, 0] = value
        self.get_frames.cache_clear()

    @lru_cache(maxsize=1)
    def get_frames(self):
        return self.ann.iloc[:, 0]

    # * ---------------------------------------------
    @property
    def ids(self):
        return self.get_ids()

    def set_ids(self, value: np.ndarray):
        self.ann.iloc[:, 1] = value
        self.get_ids.cache_clear()

    @lru_cache(maxsize=1)
    def get_ids(self):
        return self.ann.iloc[:, 1]

    # * ---------------------------------------------
    @property
    def prob(self):
        return self.get_probs()

    def set_prob(self, value: np.ndarray):
        self.ann.iloc[:, 6] = value
        self.get_probs.cache_clear()

    @lru_cache(maxsize=1)
    def get_probs(self):
        return np.array(self.ann.iloc[:, 6])

    # * ---------------------------------------------
    @property
    def cls(self):
        return self.get_cls()

    def set_cls(self, value: np.ndarray):
        self.ann.iloc[:, 7] = value
        self.get_cls.cache_clear()

    @lru_cache(maxsize=1)
    def get_cls(self):
        return np.array(self.ann.iloc[:, 7])

    # * ---------------------------------------------
    @property
    def ori_WH(self) -> Tuple[int, int]:
        return self.__ori_WH

    def get_ori_WH(self) -> Tuple[int, int]:
        return self.__ori_WH

    def set_ori_WH(self, ori_WH: Tuple[int, int], renorm=True):
        """
        renorm 是否重新归一化

        当实例已经标注为被归一化，则renorm为True时，需要先将标注恢复到原始尺寸，再根据新的尺寸进行归一化


        """
        if self.flag_norm and self.__ori_WH == (1, 1):  # 使用归一化的数据进行实例化的，但是并没有给一个合理的 ori_WH，此时传入了新的 ori_WH，只设定即可，不必进行计算
            self.__ori_WH = ori_WH
            return

        if self.flag_norm is False:
            assert renorm, "没有没归一化过，需要手动归一化，不可设置 renorm 为False"
            self.flag_norm = True

        if renorm:
            ori_xywh = deepcopy(self.ori_xywh)
            self.set_xywh(ori_xywh / [*ori_WH, *ori_WH])
            pass
        self.__ori_WH = ori_WH

    # * -------------------- 其他信息 -------------------------
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(track_nums={len(self.ann)}, ori_WH={self.__ori_WH}, \nann=\n{self.ann}, )"

    @property
    def H(self) -> int:
        return self.__ori_WH[1]

    @property
    def W(self) -> int:
        return self.__ori_WH[0]

    @property
    def ori_xywh(self) -> np.ndarray:
        return self.xywh * [*self.__ori_WH, *self.__ori_WH]

    @property
    def ori_ltrb(self) -> np.ndarray:
        return np_converter.xywh_to_ltrb(self.ori_xywh)

    @property
    def ori_ltwh(self) -> np.ndarray:
        return np_converter.xywh_to_ltwh(self.ori_xywh)

    @property
    def num_boxes(self) -> int:
        return len(self.ann)
