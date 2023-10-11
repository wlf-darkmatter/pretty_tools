"""
这里只是一个固定的使用示例，只针对MTMCT的数据读取进行配置，不做额外拓展版

在Streamlit内部，每次保存时都会从头到尾完整运行整个Python脚本。 Streamlit内部进行了大量处理来确保应用更新的效率。
"""

import streamlit as st
from pathlib import Path
from PIL import Image
import os

st.title("MTMCT Visualization")

image_box = st.empty()

if __name__ == "__main__":
    pass

    # 测试一次遍历数据集
    path_test_data = Path("/workspace/Datasets/MTMCT/DIVO/train/Circle_View/Circle_View1/imgs")
    list_imgs = sorted(os.listdir(path_test_data))
    num_imgs = len(list_imgs)
    bar = st.progress(0)
    for i, name_img in enumerate(list_imgs):
        image = Image.open(path_test_data.joinpath(name_img))
        image_box.image(image, caption="Test Image")
        bar.progress((i + 1) / num_imgs)
    st.text("Over")
