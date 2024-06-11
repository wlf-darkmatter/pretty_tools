from pretty_tools._C_pretty_tools.datastruct.misc_11 import cy_get_gt_match_from_id as get_match_id_11
from pretty_tools.datastruct._C_misc import cy_get_gt_match_from_id as get_match_id_cython
import numpy as np

def test():
    # 生成随机数据
    a = np.random.randint(0,5,size=(10,),dtype=np.longlong)
    b = np.random.randint(0,5,size=(10,),dtype=np.longlong)
    
    ret = get_match_id_11(a,b)
    ret2 = get_match_id_cython(a,b)
    print(ret)
    print(ret2)
    
if __name__=='__main__':
    test()