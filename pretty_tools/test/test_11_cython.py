import pytest
from pretty_tools._C_pretty_tools.datastruct.misc_11 import cy_get_gt_match_from_id as get_match_id_11
from pretty_tools.datastruct.misc import cy_get_gt_match_from_id
import numpy as np

def test_result():
    # 生成随机数据
    a = np.arange(10,dtype=np.int64)
    np.random.shuffle(a)
    b = np.random.choice(np.arange(10,dtype=np.int64),size=(5,),replace=False)
    
    ret = get_match_id_11(a,b)
    print(ret)
    ret2 = cy_get_gt_match_from_id(a,b)
    print(ret2)
    assert np.allclose(ret[0],ret2[0],1e-6)
    assert np.allclose(ret[1],ret2[1],1e-6)

