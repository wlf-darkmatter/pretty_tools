from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union, Optional

import numpy as np
from pandas import DataFrame
from PIL import Image

from .general_ann import GeneralAnn, track_column

# * 还是需要创建一个通用的锚框管理器，否则功能拓展太麻烦了
"""
调整引用，尽量不在脚本中引用torch，会造成一定程度的速度损失
"""


class TrackInstance:
    pass

    # todo  以后必须要转换成使用这个类记录一个跟踪对象，而且内部的锚框还要使用另一个类，因为作为拓展，可能这个锚框是3D锚框，也可能是其他格式的锚框，所以干脆使用其他类的实例作为锚框对象
    def __init__(self) -> None:
        pass


class TrackCameraInstances(GeneralAnn):
    """
    专门存储结果的一个类，这个类设计的时候只考虑到一个相机，内部包含的是一个相机内的所有目标，而非单个目标
    """

    column = track_column

    @staticmethod
    def from_general(from_general: GeneralAnn) -> TrackCameraInstances:
        return TrackCameraInstances(from_general=from_general)

    def __init__(
        self,
        ann: Union[None, np.ndarray, DataFrame] = None,
        ori_ann=None,
        str_format=None,
        ori_WH: "None | Tuple[int,int]" = None,
        ori_img: "None | Image.Image" = None,
        from_general: "None | GeneralAnn" = None,
    ) -> None:
        """
        默认存放的 str_format="xywh"
        默认的 origin_size (W,H) 为 1,1

        * data: np.ndarray, 则必须为 (N, >=6) 的格式，其中 6 为 [frame, id, xc, yc, w, h]， 其中 frame 和 id 必须为 int 类型
        """
        import torch

        if from_general is not None:
            #! 内部会要求除了 from_general 其他都是None
            super().__init__(from_general, ori_ann, str_format=str_format, ori_WH=None, ori_img=None, _move_copy=True)
            if ori_WH is not None and not self.flag_norm:
                self.set_ori_WH(ori_WH, renorm=True)
        else:
            super().__init__(ann, ori_ann, str_format=str_format, ori_WH=ori_WH, ori_img=ori_img)

        self.camera_name = None

        self.embbeding: "List[np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor | Dict[str, Any]"  # * 用于记录每个跟踪对象的其他特征

    @property
    def num_tracks(self) -> int:
        return len(self.ann)

    @property
    def range_curr_id(self) -> Tuple[int, int]:
        if self.num_tracks != 0:
            return (self.ids.min(), self.ids.max())
        else:
            return (0, 0)
