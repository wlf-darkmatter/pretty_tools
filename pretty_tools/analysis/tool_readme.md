# 性能分析工具

## 工具安装

### 逐行运行时间分析

```bash
# 在需要测试的函数加上@profile 装饰
pip install line_profiler
```

### 逐行内存分析

```bash
# 在需要测试的函数加上@profile 装饰

pip install memory_profiler
pip install psutil
```

### 可视化显示工具

```bash
pip install snakeviz
```

## 工具使用

### 内存分析

### 性能分析

```bash
python3 -m cProfile -s cumulative -o ./tmp/result.stats main.py

```

### 性能分析可视化

```bash
snakeviz ./tmp/result.stats
```

这个命令可能安装了找不到，一般会安装在 `~/.local/bin/` 下
