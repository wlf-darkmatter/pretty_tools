from typing import Any, Dict, List, Literal, Tuple, TypeVar, Union

import numpy as np
from pretty_tools.datastruct.cython_bbox import cy_bbox_overlaps_giou, cy_bbox_overlaps_iou

Optional_iou_type = Literal["giou", "iou", "diou", "ciou"]
Optional_distance_type = Literal["cosine", "minkowski", "euclidean"]
# * --------------------------------------------------------------------------------
T = TypeVar("T")


class distance_tools:
    """
    todo 计划迁移到 pretty_tools 中

    计算向量的相似度工具集

    .. note::
        **Stable** 模块，长期支持

    距离度量计算工具, 使用 **torch** 作为运行后端


    特征距离部分，目前支持设置 :code:`metric` 为以下几个:
        - 余弦距离: `cosine`
        - 闵可夫斯基距离: `minkowski`
        - 欧氏距离: `euclidean`

    """

    @staticmethod
    def calc_embedding_distance(data_a: T, data_b: T, metric: Optional_distance_type = "cosine", *args, **kwargs) -> T:
        """
        输入两组特征向量，计算特征距离矩阵，支持 numpy.ndarray 和 torch.Tensor 类型

        Args:
            data_a (np.ndarray or torch.Tensor): :math:`(n, d)` 向量
            data_b (np.ndarray or torch.Tensor): :math:`(m, d)` 向量

        .. note::
            ! 输出的距离越小越相似

        Return:
            np.ndarray or torch.Tensor: :math:`(n, m)` 距离矩阵

        """

        assert data_a.ndim == 2 and data_b.ndim == 2, "输入的数据必须是二维的"
        assert data_a.shape[1] == data_b.shape[1], "输入的深度必须是一致"
        assert data_a.__class__.__name__ == data_b.__class__.__name__, "输入的数据类型必须一致"
        if data_a.__class__.__name__ == "Tensor":
            import torch
            import torch.nn.functional as F

            if len(data_a) == 0 or len(data_b) == 0:
                return torch.zeros((len(data_a), len(data_b)), dtype=torch.float).to(data_a.device)
            if metric == "minkowski":
                dist_matrix = torch.cdist(data_a, data_b, *args, **kwargs)
            elif metric == "cosine":
                dist_matrix = 1 - F.cosine_similarity(data_a[:, None, :], data_b[None, :, :], dim=2)
            elif metric == "euclidean":
                dist_matrix = torch.cdist(data_a, data_b)  #! 注意，使用欧氏距离的时候，这里的距离范围并没有限定在 [0, 1] 之间
            else:
                raise NotImplementedError("不支持的距离度量")

        else:
            from scipy.spatial.distance import cdist

            if len(data_a) == 0 or len(data_b) == 0:
                return np.zeros((len(data_a), len(data_b)), dtype=float)

            dist_matrix = cdist(data_a, data_b, metric, *args, **kwargs)  # / 2.0  # Nomalized features # type: ignore
        return dist_matrix

    @staticmethod
    def calc_embedding_similarity(data_a: T, data_b: T, metric: Optional_distance_type = "cosine", *args, **kwargs) -> T:
        """
        输入两组特征向量，计算特征相似度矩阵，支持 numpy.ndarray 和 torch.Tensor 类型

        Args:
            data_a (np.ndarray or torch.Tensor): :math:`(n, d)` 向量
            data_b (np.ndarray or torch.Tensor): :math:`(m, d)` 向量

        .. note::
            ! 输出的亲合度越大越相似

        Return:
            np.ndarray or torch.Tensor: :math:`(n, m)` 亲合度矩阵

        """
        dis_matrix = distance_tools.calc_embedding_distance(data_a, data_b, metric, *args, **kwargs)

        return 1 - dis_matrix

    @staticmethod
    def calc_ious_overlap(ltrbs_a: Union[list[T], T], ltrbs_b: Union[list[T], T], iou_type: Optional_iou_type = "iou") -> T:
        """
        Compute cost based on IoU，暂时只支持 np.ndarray 数据类型
        :type ltrbs_a: list[ltrb] | np.ndarray
        :type ltrbs_b: list[ltrb] | np.ndarray

        :rtype ious np.ndarray
        输出的值越大，说明重合度越大

        """
        ious = np.zeros((len(ltrbs_a), len(ltrbs_b)), dtype=float)
        if ious.size == 0:
            return ious
        if iou_type == "iou":
            ious = cy_bbox_overlaps_iou(np.ascontiguousarray(ltrbs_a, dtype=float), np.ascontiguousarray(ltrbs_b, dtype=float))
        elif iou_type == "giou":
            ious = cy_bbox_overlaps_giou(np.ascontiguousarray(ltrbs_a, dtype=float), np.ascontiguousarray(ltrbs_b, dtype=float))
        else:
            raise ValueError(f"iou_type={iou_type} is not supported")
        return ious
