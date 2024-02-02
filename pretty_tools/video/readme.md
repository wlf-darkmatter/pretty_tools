# Video 读取

## 安装环境

## 测试

### 生成测试视频

### 测试环境

```
[GPU]:
Driver Version                            : 525.85.12
CUDA Version                              : 12.0
Attached GPUs                             : 2
    Product Name                          : NVIDIA GeForce RTX 3090
    Product Name                          : NVIDIA GeForce RTX 3090

[CPU]:
Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz
Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz

[RAM]
MemTotal:       263773100 kB (251.55 GB)

```

```yaml
Test1:
    Deffcode:
        config:
         - cuda=True
         - ffmpeg=True
         - num_thread=0
         - num_cache=0
         - type=numpy
        result: 730.1600363254547
Test2:
    VPF: # 仅用单卡
        config:
         - cuda=True
         - ffmpeg=False
         - num_thread=0
         - num_cache=0
         - type=numpy
         - gpu_id=0  ☆
        result: 287.78179240226746
    VPF: # 循环使用多卡 更慢
        config:
         - cuda=True
         - ffmpeg=False
         - num_thread=0
         - num_cache=0
         - type=numpy
         - gpu_id=-1 ☆
        result: 476.8549120426178
        memory:
         - 1226MiB
         - 1226MiB
Test3:
    # 都在gpu上存储，比较两个显卡通道的耗时差异
    VPF: # 使用单卡，但是走的是cuda转torch
        config:
         - cuda=True
         - ffmpeg=False
         - num_thread=0
         - num_cache=0
         - type=torch
         - gpu_id=0 ☆
        result: 164.4672737121582
    VPF: # 使用单卡，但是走的是cuda转torch
        config:
         - cuda=True
         - ffmpeg=False
         - num_thread=0
         - num_cache=0
         - type=torch
         - gpu_id=1 ☆
        result: 164.70146870613098
    VPF: # 使用单卡，但是走的是cuda转torch
        config:
         - cuda=True
         - ffmpeg=False
         - num_thread=0
         - num_cache=0
         - type=torch
         - gpu_id=-1 ☆
        result: 87.04115009307861
        memory:
         - 1226MiB
         - 1226MiB

```

## api 文档

### 数据格式

```yaml
Backend_Decode_Base:
  shape: [h, w, c]
  numpy_shape: [h, w, c]
  torch_shape: [c, h, w]
```

### Video Processing Framework

```python
import PyNvCodec as nvc
# 这将创建一个在gpu上的解码器实例
nv_dmx = nvc.PyNvDecoder(path_video, gpu_id)

# 下面的操作会在GPU上进行解码，解码的是视频的下一帧，并且将解码后的数据返回到主机内存中
frame_nv12 = np.ndarray(shape=(0), dtype=np.uint8) #一个存放解码后数据的数组
packet = np.ndarray(shape=(0), dtype=np.uint8) # 存放的是未解码的一个帧包， 这个在使用ffmpeg解码的时候才会用到
packet_data = nvc.PacketData() # 同样存放未解码的帧包
nv_dmx.DecodeSingleFrame(frame_nv12, packet_data) # 进行解码，解码成功后会返回 True, 解码后的图像数据在frame_nv12中


```
