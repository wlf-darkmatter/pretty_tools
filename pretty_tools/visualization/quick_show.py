from typing import (Any, Generic, Iterable, Optional, Sequence, Tuple, TypeVar,
                    Union)

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy import sparse

array_Type = Union[np.ndarray, sparse.spmatrix]


def quick_show_array(input_array: array_Type):
    """
    因为大型矩阵要查看值非常麻烦，这里直接提供一个快速可视化的调用
    """
    import seaborn as sns
    from matplotlib import pyplot as plt

    fig = plt.figure()
    if isinstance(input_array, sparse.spmatrix):
        sns.heatmap(input_array.todense(), linewidths=0.1, square=True, ax=fig.gca())
    else:
        sns.heatmap(input_array, linewidths=0.1, square=True, ax=fig.gca())
    fig.gca().xaxis.set_ticks_position("top")  # * 将x轴的位置设置在顶部
    fig.gca().invert_yaxis()  # * y轴反向
    return fig


def quick_show_edgeindex(row, col):
    """
    因为大型矩阵要查看值非常麻烦，这里直接提供一个快速可视化的调用,
    输入的是矩阵的row和col
    """
    import seaborn as sns
    from matplotlib import pyplot as plt
    from pandas import DataFrame

    fig = plt.figure()
    data = DataFrame(np.vstack([row, col, np.ones(len(col))]).T, columns=["i", "j", "v"])

    sns.scatterplot(data, x="j", y="i", size="v", ax=fig.gca())
    fig.gca().xaxis.set_ticks_position("top")  # * 将x轴的位置设置在顶部
    fig.gca().invert_yaxis()  # * y轴反向
    return fig
