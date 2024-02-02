from typing import Optional, Tuple, Union

import numpy as np
from scipy import sparse


def match_result_check(edge_index_gt: np.ndarray, edge_index_pred: np.ndarray):
    """
    注意: edge_index_pred 可能 shape 为 (2, 0)
    Return:
        edge_index_TP, edge_index_FN, edge_index_FP
    """
    max_merge = np.concatenate((edge_index_gt, edge_index_pred), axis=1)
    if max_merge.shape[1] == 0:
        edge_index_TP = np.zeros((2, 0), dtype=np.int32)
        edge_index_FN = np.zeros((2, 0), dtype=np.int32)
        edge_index_FP = np.zeros((2, 0), dtype=np.int32)
    else:
        max_ = max_merge.max() + 1
        adj_gt = sparse.coo_matrix((np.ones(edge_index_gt.shape[1]), edge_index_gt), shape=(max_, max_))
        adj_pred = sparse.coo_matrix((np.ones(edge_index_pred.shape[1]), edge_index_pred), shape=(max_, max_))
        adj_sum, adj_diff = adj_gt + adj_pred, adj_gt - adj_pred

        edge_index_TP = np.stack((adj_sum == 2).nonzero())  # type: ignore
        edge_index_FN = np.stack((adj_diff == 1).nonzero())  # type: ignore
        edge_index_FP = np.stack((adj_diff == -1).nonzero())  # type: ignore

    return edge_index_TP, edge_index_FN, edge_index_FP
