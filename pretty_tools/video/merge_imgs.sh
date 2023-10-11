#!/bin/bash

#* 用ffmpeg将图片合成视频
ffmpeg -f image2 -i %08d.jpg -vcodec libx264 -r 25 -pix_fmt yuv420p out.mp4


#* 通过gpu加速 使用ffmpeg合成图像为视频
ffmpeg -f image2 -i %08d.jpg -vcodec libx264 -r 25 -pix_fmt yuv420p -hwaccel cuvid out.mp4

#* 循环遍历文件夹下的所有图片
for i in $(ls);do
    if [ -d $i ];then
        # -y 覆盖输出文件
        ffmpeg -y -f image2 -i $i/%08d.jpg -vcodec libx264 -r 12 -pix_fmt yuv420p $i.mp4
    fi
done


#* （使用cuda加速）循环遍历文件夹下的所有图片
for i in $(ls);do
    if [ -d $i ];then
        # -y 覆盖输出文件
        ffmpeg  -y -f image2 -i $i/%08d.jpg -vcodec h264_nvenc -b:v 10m -r 12 -pix_fmt yuv420p $i.mp4
    fi
done