"""
Multi Index Dict
"""
from __future__ import annotations

from typing import (Any, Callable, Generic, Hashable, Iterable, Iterator, List,
                    Optional, Sequence, Tuple, TypeVar, Union)

import numpy as np
from pandas import DataFrame
from scipy import sparse

KT = TypeVar("KT")
VT = TypeVar("VT")


class mdict(Generic[KT, VT]):
    """
    多重索引本身是通过调用了 :class:`DataFrame` 来实现的

    .. code-block:: python

        x = mdict() # 创建一个二重索引字典（默认）
        a = mdict(3) # 创建一个三重索引字典
        b = mdict(5) # 创建一个五重索引字典

    对字典进行赋值

    .. code-block:: python

        a[1, 2, '3'] = "example"
        a['1', 4, '3'] = "example"
        a[0, 0, 0] = 0
        a[1, 2, '3'] == a['1', 4, '3']
        >>> True

        a[0, 0, 0] == a[1, 2, '3']
        >>> False


    判断多重索引字典中是否有指定的索引

    .. code-block:: python

        [1, 2, '3'] in a
        >>> True

        ['3', 2, 1] in a
        >>> True

        ["x", "x", 'y'] in a
        >>> False

    """

    def __init__(self, dim: int = 2) -> None:
        self.dim = dim
        assert dim >= 2, "dim must >= 2"
        self.__main_dict: dict[Any, VT] = {}
        self.__index_dict: dict[Any, KT] = {}  # * 根据给入的 str_name，获取 index

        # * -------------- 索引名称只需要在这里更改即可 ------------------
        self.__index_columns: List[str] = [f"index{i}" for i in range(self.dim)]
        self.__dataframe_columns = self.__index_columns + ["index_name"]
        self.dataframe = DataFrame(columns=self.__dataframe_columns)
        self.dataframe.set_index(self.__index_columns)

        self.__func_name = lambda tuple_index: ";".join(sorted([str(type(x))[7:] + str(x) for x in tuple_index]))
        self.__repr_status = True
        self.__str_dict = {}

        self.__dict_items: dict[Hashable, int] = {}  # * items 引用计数器

    def __getitem__(self, *args: Sequence[Hashable]) -> VT:
        # todo 这里之后再改，如果dim为3，输入了 一个索引，则应当返回一个 dim为2的MDict；输入了2个索引，则应该返回一个dict，输入了3个索引，则应该返回一个对象
        assert len(args[0]) <= self.dim, f"Required dim {len(args)} bigger than MDict dim {self.dim}"
        index_name = self.__func_name(args[0])
        try:
            return self.__main_dict[index_name]
        except KeyError:
            raise KeyError(args[0])

    def setdefault(self, *args):
        index_args: "Hashable | Sequence[Hashable]" = args[0]
        target = args[1]
        if index_args not in self:
            self[index_args] = target

    def __setitem__(self, *args):
        import pandas as pd

        index_args: "Hashable | Sequence[Hashable]" = args[0]
        target = args[1]

        # todo 暂时不支持批量赋值
        if isinstance(index_args, Sequence):
            assert len(index_args) == self.dim, f"Required dim {len(index_args)} bigger than MDict dim {self.dim}"
            for i in index_args:
                # * 添加引用计数器
                self.__dict_items.setdefault(i, 0)
                self.__dict_items[i] += 1
        else:
            assert self.dim == 1
            self.__dict_items.setdefault(index_args, 0)
            self.__dict_items[index_args] += 1

        self.__repr_status = False
        #! 想要快就用C++写
        index_name = self.__func_name(index_args)  # todo 这里使用了字符串处理以及 sorted，通过python的for循环遍历的时候效率并不会很低，因为for循环本身效率就很低
        tmp_dataframe = pd.DataFrame(data=[[*index_args, index_name]], columns=self.__dataframe_columns)
        self.dataframe = pd.concat([self.dataframe, tmp_dataframe], axis=0)
        self.dataframe.set_index(self.__index_columns)

        self.__main_dict[index_name] = target
        self.__index_dict[index_name] = index_args
        pass

    def __delitem__(self, *args: "Hashable | Sequence[Hashable]"):
        # todo 暂时不支持批量删除
        index_args = args[0]  #! 这句话将该方法设置为了只能处理一个批次

        self.__repr_status = False
        index_name = self.__func_name(index_args)
        del self.__main_dict[index_name]
        del self.__index_dict[index_name]

        if isinstance(index_args, Sequence):
            assert len(index_args) == self.dim, f"Required dim {len(index_args)} bigger than MDict dim {self.dim}"
            for i in index_args:
                # * 添加引用计数器
                self.__dict_items[i] -= 1
                if self.__dict_items[i] == 0:
                    del self.__dict_items[i]
        else:
            assert self.dim == 1
            self.__dict_items.setdefault(index_args, 0)
            self.__dict_items[index_args] += 1
            if self.__dict_items[index_args] == 0:
                del self.__dict_items[index_args]
        # todo dataframe 中仍有残留，这里就暂时不处理了

    def __repr__(self) -> str:
        if not self.__repr_status:
            self.__repr_status = True
            self.__str_dict = "{\n"
            for k in self.__main_dict.keys():
                self.__str_dict += f"{self.__index_dict[k]}: " + str(self.__main_dict[k]) + ",\n"
            self.__str_dict += "}"

        return f"dim: {self.dim}, len: {len(self.__main_dict)}\n{self.__str_dict}"

    def __len__(self) -> int:
        return len(self.__main_dict)

    def __contains__(self, *args) -> bool:
        """
        ! 需要注意一种情况，即内部的格式是np的 int64或者其他类型的时候，无法和内置的python的int float类型的判定为同一个类型，进而可能会出现两个相同的键
        """
        if isinstance(args, Tuple) and len(args) == 1:
            args = args[0]
        index_name = self.__func_name(args)
        if index_name in self.__main_dict:
            return True
        else:
            return False

    def items(self) -> Iterable[tuple[KT, VT]]:
        for k, v in self.__main_dict.items():
            yield self.__index_dict[k], v

    def __iter__(self):
        raise NotImplementedError("mdict 的迭代器还没有实现，即时实现，也应当是一个只给出索引信息的迭代器")

    def keys(self) -> Iterable[KT]:
        return self.__index_dict.values()

    def str_keys(self) -> Iterable[str]:
        return self.__index_dict.keys()

    def values(self) -> Iterable[VT]:
        return self.__main_dict.values()

    @property
    def loc(self):
        UserWarning("这个是Mdict为了和DataFrame互通的一个属性，尽量不要使用它")
        return self

    @property
    def num_items(self) -> int:  #! 这里的 num_ites 是一个特殊的计数器，记录了所有索引对象的个数，详细查看测试环节
        return len(self.__dict_items)

    @staticmethod
    def combinations(iterable: Iterable, calc_fn: Callable, ndim=1):
        r"""
        如果没有输入 ndim，则会自动判断这个函数需要多少个参数，然后遍历所有的排列组合，将结果存储在一个 MDict 中


        Note
        ----

        #! 注意，遍历的时候，其顺序是从前往后的排列组合，即所有组合情况中，计算形式为

        .. math:: y_i =\text{calc_fn} (x_i), (i\le n )


        当 :obj:`len(iterable) = n, ndim = m`，则输出的 mdict 的长度为 :math:`|C_n^m|`

        :obj:`iterable` 应当是一个迭代对象，而不是一个迭代器
        """
        import inspect
        from itertools import combinations

        if ndim == 1:
            # 自动判断函数需要几个参数
            ndim = len(inspect.signature(calc_fn).parameters)
            assert ndim >= 1
        x = mdict(ndim)
        if type(iterable) is dict:
            target_item = iterable
            list_index = list(iterable.keys())
        else:
            target_item = list(iterable)
            list_index = range(len(target_item))

        for param_index in combinations(list_index, ndim):
            x[param_index] = calc_fn(*[target_item[i] for i in param_index])
        return x

    def apply(self, calc_fn: Callable) -> mdict:
        """
        对内部的所有对象都进行处理，

        Parameters
        ----------

        calc_fn: function
            - :obj:`calc_fn(v)` # 只有一个参数，则该参数应当是 mdict 的值
            - :obj:`calc_fn(v, k)` # 如果能接受两个参数，则第二个参数会是mdict内部对象的索引号(Tuple类型)

        """
        import inspect

        n_args = len(inspect.signature(calc_fn).parameters)
        if n_args == 1:
            for k, v in self.items():
                self[k] = calc_fn(v)
        elif n_args == 2:
            for k, v in self.items():
                self[k] = calc_fn(v, k)
        else:
            raise TypeError("calc_fn 需要的参数应当为 1 或者 2 个")

        return self

    @classmethod
    def from_sparse(cls, index_slice: Union[np.ndarray, list[int]], sparse_array: sparse.spmatrix, mdict_input: Optional[mdict] = None) -> mdict:
        """
        将 sparse 的信息转换成 :class:`mdict`
        默认输出的是 二维的 :class:`mdict`

        :class:`mdict` 的格式必定为 :code:`mdict[index_a, index_b] = (inner_index_array, data_array)`

        Args:
            index_a (int)
            index_b (int)
            inner_index_array (list[list[int]]): shape is like :code:`(n, 2)`
            data_array (list[Any]): shape is like :code:`(n, )`

        Note
        ----

        注意，如果传入了 mdict_input 则会对其进行修改，这里使用的是引用，而不是拷贝

        """
        from pretty_tools.datastruct import np_enhance

        if mdict_input is None:
            mdict_input = mdict(2)

        sparse_array = sparse_array.tocoo()
        a, b = np_enhance.index_convert_to_block([index_slice, index_slice], [sparse_array.row, sparse_array.col])
        for block_i, block_j, inner_i, inner_j, data in zip(*a, *b, sparse_array.data):
            mdict_input.setdefault((int(block_i), int(block_j)), ([], []))
            mdict_input[(int(block_i), int(block_j))][0].append([int(inner_i), int(inner_j)])
            mdict_input[(int(block_i), int(block_j))][1].append(data)
        return mdict_input

    @classmethod
    def to_sparse(cls, index_slice: Union[np.ndarray, list[int]], mdict_input: mdict, sparse_array: Optional[sparse.spmatrix] = None) -> sparse.lil_matrix:
        """
        Args:
            index_slice :
            mdict_input (mdict): 格式必定为 :code:`mdict[index_a, index_b] = (inner_index_array, data_array)`
            sparse_array: 如果给入了 sparse_array，则会在其 **拷贝上** 进行修改，否则会新建一个

        :return:
        """
        assert index_slice[0] == 0, "index_slice 的首个元素必须是0"
        if sparse_array is not None:
            out_sparse = sparse_array.tolil()
        else:
            out_sparse = sparse.lil_matrix((index_slice[-1], index_slice[-1]))
        for index, matched in mdict_input.items():
            if len(matched[0]) > 0:
                result_pos = matched[0] + index_slice[*[index]]  # 格式是 ij
                out_sparse[*result_pos.T] = matched[1]
        return out_sparse
