import numpy as np
import pytest
from pretty_tools.metrics import MatchMetrics


class Test_Match:
    def setup_method(self):
        self.match_metrics = MatchMetrics(["Precision", "Recall", "TP", "FP", "FN", "F1"])
        self.edge_index_gt = np.array([[3, 0, 3, 10, 12], [10, 13, 14, 14, 20]])
        self.edge_index_pred = np.array([[2, 3, 4, 0, 3, 4, 6, 7, 8, 10, 11, 12, 6, 7], [9, 10, 11, 13, 14, 15, 16, 20, 16, 14, 15, 20, 8, 12]])

        self.match_metrics.load_gt_edge_index(self.edge_index_gt)
        self.match_metrics.load_pred_edge_index(self.edge_index_pred)

    def test_calc_TP(self):
        self.match_metrics.calc_TP()
        getattr(self.match_metrics, "TP") == 5

    def test_calc_FP(self):
        self.match_metrics.calc_FP()
        getattr(self.match_metrics, "FP") == 9

    def test_calc_FN(self):
        self.match_metrics.calc_FN()
        getattr(self.match_metrics, "FN") == 0

    # todo 还有一些指标的计算没有进行验证，而且数据量也不够大，以后再改



if __name__ == "__main__":
    pytest.main(
        [
            "-s",
            "-l",
            "test_metrics.py",
        ]
    )
