import numpy as np


def np_Max_error(a, b):
    return np.max(np.abs(a - b))


error_fp128 = 2**-112
error_fp64 = 2**-52
error_fp32 = 2**-23
error_fp16 = 2**-10
error_fp8 = 2**-4
