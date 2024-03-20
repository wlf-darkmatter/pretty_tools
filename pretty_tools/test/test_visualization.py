import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
from PIL import Image
from pretty_tools.datastruct import mdict
from pretty_tools.visualization import Pretty_Draw, Visiual_Tools

# fmt: off
path_graph_0 = Path(__file__).parent.with_name("resources") / "data" / "floor1_00000001.pkl"
path_graph_1 = Path(__file__).parent.with_name("resources") / "data" / "floor2_00000001.pkl"
path_graph_2 = Path(__file__).parent.with_name("resources") / "data" / "floor3_00000001.pkl"

path_image_0 = Path(__file__).parent.with_name("resources") / "imgs" / "Floor_View1_000001.jpg"
path_image_1 = Path(__file__).parent.with_name("resources") / "imgs" / "Floor_View2_000001.jpg"
path_image_2 = Path(__file__).parent.with_name("resources") / "imgs" / "Floor_View3_000001.jpg"

path_tmp = Path(__file__).with_name("tmp")
path_tmp.mkdir(exist_ok=True)


def np_AvgR_error(a, b):
    return np.abs(np.sum(a - b)) / np.sum(b) / b.size


def np_Max_error(a, b):
    return np.max(np.abs(a - b))


def get_url_str(url):
    import urllib.request

    req = urllib.request.Request(url)
    resp = urllib.request.urlopen(req)
    data = resp.read().decode("utf-8")
    return data


"""
误差估计 二进制的科学计数法最左边部分总是1
float128 尾数112 指数15
float64 尾数52 指数11
float32 尾数23 指数8
float16 尾数10 指数5
float8  尾数4  指数2


"""
error_fp128 = 2**-112
error_fp64 = 2**-52
error_fp32 = 2**-23
error_fp16 = 2**-10
error_fp8 = 2**-4



# fmt: off



def quick_save(images: "List[Image.Image | np.ndarray] | Image.Image | np.ndarray", path_save: "str | Path"):
    from matplotlib import pyplot as plt

    if isinstance(images, list):
        fig, axs = plt.subplots(1, len(images))
        for i, img in enumerate(images):
            axs[i].imshow(img)
    else:
        fig = plt.figure()
        fig.gca().imshow(images)
    fig.savefig(path_save)


class Test_Draw:
    def setup_method(self):
        from pretty_tools.datastruct import (TrackCameraGraph,
                                             TrackCameraInstances)
        from pretty_tools.datastruct.track_graph import TrackCameraGraph

        self.tg0 = TrackCameraGraph.load(path_graph_0)
        self.tg1 = TrackCameraGraph.load(path_graph_1)
        self.tg2 = TrackCameraGraph.load(path_graph_2)
        self.img0 = Image.open(path_image_0)
        self.img1 = Image.open(path_image_1)
        self.img2 = Image.open(path_image_2)
        self.list_tg = [self.tg0, self.tg1, self.tg2]
        self.list_img = [self.img0, self.img1, self.img2]
        import torch.nn.functional as F


        self.dict_images = {k: v for k, v in zip(range(len(self.list_img)), self.list_img)}
        self.dict_feature = {k: v.x for k, v in zip(range(len(self.list_tg)), self.list_tg)}
        self.dict_ids = {k: np.array(v.ids) for k, v in zip(range(len(self.list_tg)), self.list_tg)}
        self.dict_ltrb = {k: np.array(v.ori_ltrb) for k, v in zip(range(len(self.list_tg)), self.list_tg)}
        fn = lambda feat_i, feat_j: 1 - F.cosine_similarity(feat_i.unsqueeze(1), feat_j.unsqueeze(0), dim=2)
        self.mdict_similarity = mdict.combinations(self.dict_feature, fn)


    def test_cut_bbox(self):
        n = self.tg0.num_nodes
        # * 切分 Image
        list_cut = Visiual_Tools.cut_bbox(self.img0, self.tg0.ori_ltwh, str_format="ltwh")
        assert len(list_cut) == n
        quick_save(list_cut, path_tmp.joinpath("test_cut_bbox_Image.jpg"))
        # * 切分 np.ndarray
        list_cut = Visiual_Tools.cut_bbox(np.array(self.img0), self.tg0.ori_ltwh, str_format="ltwh")
        assert len(list_cut) == n
        quick_save(list_cut, path_tmp.joinpath("test_cut_bbox_Numpy.jpg"))
        # return

        # * ------------------- 测试 速度 -------------------
        import time

        num_test = 10000
        # ------------------- 测试 time_cut_Image -------------------
        time_start = time.time()
        for i in range(num_test):
            list_cut = Visiual_Tools.cut_bbox(self.img0, self.tg0.ori_ltwh, str_format="ltwh")
        time_cut_Image = time.time() - time_start
        # ------------------- 测试 time_cut_Numpy -------------------
        tmp_img = np.array(self.img0)
        time_start = time.time()
        for i in range(num_test):
            list_cut = Visiual_Tools.cut_bbox(tmp_img, self.tg0.ori_ltwh, str_format="ltwh")
        time_cut_Numpy = time.time() - time_start

        with open(path_tmp.joinpath("test_cut_bbox.log"), "w") as f:
            f.write("Visiual_Tools.cut_bbox() 速度测试结果\n")
            f.write("---------------------------------------------------\n")
            f.write(f"切分 Image.Image {num_test}次， 耗时 {time_cut_Image:0.4}s\n")
            f.write(f"切分 np.ndarray  {num_test}次， 耗时 {time_cut_Numpy:0.4}s")
        pass

    def test_draw_affinity(self):
        # * 测试 字典模式的绘制
        from pretty_tools.datastruct.misc import cy_get_gt_match_from_id

        gt_match_fn = lambda ids_a, ids_b: cy_get_gt_match_from_id(ids_a, ids_b)[0]
        mdict_pair = mdict.combinations(self.dict_ids, gt_match_fn)
        fig = Pretty_Draw.draw_affinity(self.mdict_similarity, self.dict_images, self.dict_ltrb)
        fig.savefig(path_tmp.joinpath("test_draw_affinity_with_images.jpg").__str__())

        fig = Pretty_Draw.draw_affinity(self.mdict_similarity, self.dict_images, self.dict_ltrb, mdict_pair=mdict_pair)
        fig.savefig(path_tmp.joinpath("test_draw_affinity_with_images_and_matched.jpg").__str__())

    def test_draw_bboxes(self):
        image = Pretty_Draw.draw_bboxes(self.img0, self.tg0.ori_ltrb, self.tg0.ids)
        image.save(path_tmp.joinpath("test_draw_bboxes.jpg").__str__())

        # --- 测试 matplotlib 可视化功能 ---

    def test_draw_bboxes_with_plt(self):
        fig = Pretty_Draw.draw_bboxes_matplotlib(self.img0, self.tg0.ori_ltrb, self.tg0.ids, str_format="ltrb")


if __name__ == "__main__":
    pytest.main(
        [
            "-s",
            "-l",
            "test_visualization.py",
        ]
    )
