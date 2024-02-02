import numpy as np


def get_clear_foreground_color(background_color: np.ndarray, thresh=0.5) -> np.ndarray:
    assert background_color.ndim == 2
    assert background_color.shape[1] == 3
    # 计算与黑色的对比度
    gray = np.sum(background_color * [0.299, 0.587, 0.114], axis=1)
    output = np.zeros_like(gray)
    output[gray > thresh] = 0
    output[gray < thresh] = 1
    return output
