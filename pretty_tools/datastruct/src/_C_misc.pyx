# cython:language_level=3
# --------------------------------------------------------
# ContionTrack
# Copyright (c) 2023 BIT
# Licensed under The MIT License [see LICENSE for details]
# Written by Lingfeng Wang
# --------------------------------------------------------
cimport cython #! 这个不能少
import numpy as np
cimport numpy as cnp
# Cython不允许一个cdef定义的函数被外部python代码直接调用

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_get_gt_match_from_id(        
    cnp.ndarray[cnp.int64_t, ndim=1] np_id_a,
    cnp.ndarray[cnp.int64_t, ndim=1] np_id_b,
    ):
    cdef int i, j
    cdef unsigned int N = np_id_a.shape[0]
    cdef unsigned int M = np_id_b.shape[0]
    cdef unsigned int K = max(N, M)
    cdef unsigned int count = 0

    cdef cnp.ndarray[cnp.int64_t, ndim=2] matched = np.zeros((K, 3), dtype=np.int_)


    for i in range(N):
        for j in range(M):
            if np_id_a[i] == np_id_b[j]:
                matched[count][0] = i
                matched[count][1] = j
                matched[count][2] = np_id_a[i]
                count += 1
                continue
    return matched[:count, :2], matched[:count, 2]