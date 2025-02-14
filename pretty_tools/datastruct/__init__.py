"""
定义了诸多的数据结构，同时对python、numpy、torch的一些数据操作设计了快速实现

"""

from . import X_bbox
from .X_bbox import ltrb_to_ltwh as np_ltrb_to_ltwh
from .X_bbox import ltrb_to_xywh as np_ltrb_to_xywh
from .X_bbox import ltwh_to_ltrb as np_ltwh_to_ltrb
from .X_bbox import ltwh_to_xywh as np_ltwh_to_xywh
from .X_bbox import xywh_to_ltrb as np_xywh_to_ltrb
from .X_bbox import xywh_to_ltwh as np_xywh_to_ltwh
from .color_convert import *
from .general_ann import GeneralAnn
from .multi_index_dict import mdict
from .track_graph import TrackCameraGraph
from .track_instance import TrackCameraInstances, TrackInstance


try:
    from . import cython_bbox
except:
    print("pretty_tools/datastruct/__init__.py: cython_bbox not found, please compile it first")

__all__ = [
    "mdict",
    "graph_enhance",
    "torch_enhance",
    "np_enhance",
]
