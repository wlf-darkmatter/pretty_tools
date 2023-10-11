# 使用说明

文档更新必定落后于代码，当前的文档说明适配版本号为 `0.1.7`

## 安装

```bash
# 方法 1
pip install -e pretty_tools
# 方法 2
pip install git+http://git.x-contion.top/lab_share/pretty_tools.git

```

## 使用

```bash
import pretty_tools

```

# 功能文档

## data_struct

### multi_index_dict

```python
from pretty_tools.data_struct import mdict
# 多重索引本身是通过调用了 DataFrame 来实现的


x = mdict() # 创建一个二重索引字典（默认）
a = mdict(3) # 创建一个三重索引字典
b = mdict(5) # 创建一个五重索引字典

len(a) # 输出字典的长度，即字典中的元素个数

# 对字典进行赋值
a[1, 2, '3'] = "example"
a['1', '2', '3'] = "example"
a[0, 0, 0] = 0
a[1, 2, '3'] == a['1', '2', '3']
>>> True
a[0, 0, 0] == a[1, 2, '3']
>>> False


# 判断多重索引字典中是否有指定的索引


[1, 2, '3'] in a
>>> True
['3', 2, 1] in a
>>> True
["x", "x", 'y'] in a
>>> False

```
