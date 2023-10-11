"""
统计匹配的精度
"""
from typing import LiteralString, Literal, List, Union, Tuple, Optional, Callable, Generic, TypeVar
import numpy as np
from scipy import sparse
import rich

Type_Match_Metrics = Literal["Precision", "Recall", "F1", "TP", "FP", "FN"]
T_M = TypeVar("T_M", bound=Type_Match_Metrics)  #! 这里用一个 bound 就可以根据 bound 的类型推断出 T_M 的类型了，也就是类似 pandas 的那种列名称提示
dict_input_fn = {}


def cache_need(str_outname, list_metrics: Optional[List[str]] = None):
    def outwrapper(fn: Callable):
        def wrapper(_self):
            self: MatchMetrics = _self
            if self.debug:
                rich.print(f"Calc cache ['{str_outname}'] by fn {fn.__name__}(), using input: {list_metrics}")

            list_args = [getattr(self, i) if hasattr(self, i) else dict_input_fn[i](self) for i in list_metrics]
            result = fn(self, *list_args)
            setattr(self, str_outname, result)
            self.dict_final_metrics[str_outname] = result
            return result

        dict_input_fn[str_outname] = wrapper

        #
        return wrapper

    return outwrapper


class MatchMetrics(Generic[T_M]):
    def __init__(self, list_metrics: List[T_M], debug=False) -> None:
        self.list_metrics = list_metrics
        self.debug = debug

        self.pred_score: np.ndarray

        self.adj_sum: sparse.spmatrix
        self.adj_diff: sparse.spmatrix

        self.adj_gt: sparse.spmatrix = None  # type: ignore
        self.adj_pred: sparse.spmatrix = None  # type: ignore

        self.dict_final_metrics = {}
        self.m = 0
        self.n = 0

    def __reset_shape(self):
        if self.adj_gt is not None:
            self.m = max(self.m, self.adj_gt.shape[0])
            self.n = max(self.n, self.adj_gt.shape[1])
        if self.adj_pred is not None:
            self.m = max(self.m, self.adj_pred.shape[0])
            self.n = max(self.n, self.adj_pred.shape[1])
        # * 调整尺寸
        if self.adj_pred is not None:
            if ((self.m, self.n)) != self.adj_pred.shape:
                new_adj = sparse.lil_matrix((self.m, self.n))
                m, n = self.adj_pred.shape
                new_adj[:m, :n] = self.adj_pred
                self.adj_pred = new_adj.tocoo()
        if self.adj_gt is not None:
            if ((self.m, self.n)) != self.adj_gt.shape:
                new_adj = sparse.lil_matrix((self.m, self.n))
                m, n = self.adj_gt.shape
                new_adj[:m, :n] = self.adj_gt
                self.adj_gt = new_adj.tocoo()
        pass

    def __getitem__(self, metric_name: T_M):
        if not hasattr(self, metric_name):
            dict_input_fn[metric_name](self)
        return getattr(self, metric_name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(list_metrics={self.list_metrics})"

    def calc_all_metrics(self) -> dict[T_M, Union[float, int]]:
        dict_metrics = {}
        for i in self.list_metrics:
            dict_metrics[i] = self[i]
        return dict_metrics

    def cache_clear(self):
        for i in self.list_metrics:
            if hasattr(self, i):
                delattr(self, i)
            if i in self.dict_final_metrics:
                del self.dict_final_metrics[i]

    def load_gt_edge_index(self, edge_index_gt, shape: Optional[tuple[int, int]] = None):
        if edge_index_gt.shape[1] == 0:
            adj = sparse.coo_matrix((0, 0), shape=shape)
        else:
            adj = sparse.coo_matrix((np.ones(edge_index_gt.shape[1], dtype=int), edge_index_gt), shape=shape)

        self.load_gt_adj(adj)

    def load_gt_adj(self, adj_gt):
        self.adj_gt = adj_gt
        self.cache_clear()  # * 如果发生变化，则所有结果都要重新计算，因此需要删除掉所有元素
        self.__reset_shape()

    def load_pred_edge_index(self, edge_index_pred, scores: Optional[np.ndarray] = None, shape: Optional[tuple[int, int]] = None):
        if scores is None:
            scores = np.ones(edge_index_pred.shape[1], dtype=int)  #

        if edge_index_pred.shape[1] == 0:
            adj = sparse.coo_matrix((0, 0), shape=shape)
        else:
            adj = sparse.coo_matrix((scores, edge_index_pred), shape=shape)
        self.load_pred_adj(adj)

    def load_pred_adj(self, adj_pred):
        self.adj_pred = adj_pred
        self.cache_clear()  # * 如果发生变化，则所有结果都要重新计算，因此需要删除掉所有元素
        self.__reset_shape()

    def load_pred_score(self, scores):  # *
        assert len(scores) == self.adj_pred.nnz
        raise NotImplementedError("功能还没有补全，先留白")

    @cache_need("adj_sum", ["adj_gt", "adj_pred"])
    def calc_adj_sum(self, *args):
        return args[0] + args[1]

    @cache_need("adj_diff", ["adj_gt", "adj_pred"])
    def calc_adj_diff(self, *args):
        return args[0] - args[1]

    @cache_need("TP", ["adj_sum"])
    def calc_TP(self, *args):
        adj_sum = args[0]
        return (adj_sum == 2).nnz

    @cache_need("FN", ["adj_diff"])
    def calc_FN(self, *args):  # * 漏检
        adj_diff = args[0]
        return (adj_diff == 1).nnz

    @cache_need("FP", ["adj_diff"])
    def calc_FP(self, *args):  # * 误检
        adj_diff = args[0]
        return (adj_diff == -1).nnz

    @cache_need("Recall", ["TP", "FN"])
    def calc_R(self, *args):
        TP, FN = args
        if TP == 0:
            return 0
        return TP / (TP + FN)

    @cache_need("Precision", ["TP", "FP"])
    def calc_P(self, *args):
        TP, FP = args
        if TP == 0:
            return 0
        return TP / (TP + FP)

    @cache_need("F1", ["Precision", "Recall"])
    def calc_F1(self, *args):
        P, R = args
        if P == 0 or R == 0:
            return 0
        return (2 * P * R) / (P + R)


if __name__ == "__main__":
    match_metrics = MatchMetrics(["Precision", "Recall", "TP", "FP", "FN", "F1"], debug=True)
    edge_index = np.array([[3, 0, 3, 10, 12], [10, 13, 14, 14, 20]])
    edge_index_pred = np.array([[2, 3, 4, 0, 3, 4, 6, 7, 8, 10, 11, 12, 6, 7], [9, 10, 11, 13, 14, 15, 16, 20, 16, 14, 15, 20, 8, 12]])

    match_metrics.load_gt_edge_index(edge_index)
    match_metrics.load_pred_edge_index(edge_index_pred)

    print(match_metrics["Recall"])
