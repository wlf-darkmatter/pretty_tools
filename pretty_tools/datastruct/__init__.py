"""
定义了诸多的数据结构，同时对python、numpy、torch的一些数据操作设计了快速实现

"""

from .general_ann import GeneralAnn

from .graph_enhance import merge_affinity
from .multi_index_dict import mdict
from .track_graph import TrackCameraGraph
from .track_instance import TrackCameraInstances, TrackInstance


from . import bbox_convert_np
from .bbox_convert_np import ltrb_to_ltwh as np_ltrb_to_ltwh
from .bbox_convert_np import ltrb_to_xywh as np_ltrb_to_xywh
from .bbox_convert_np import ltwh_to_ltrb as np_ltwh_to_ltrb
from .bbox_convert_np import ltwh_to_xywh as np_ltwh_to_xywh
from .bbox_convert_np import xywh_to_ltrb as np_xywh_to_ltrb
from .bbox_convert_np import xywh_to_ltwh as np_xywh_to_ltwh
from .color_convert import *

__all__ = [
    "mdict",
    "graph_enhance",
    "torch_enhance",
    "np_enhance",
]
