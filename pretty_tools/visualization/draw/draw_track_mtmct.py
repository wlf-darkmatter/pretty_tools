"""

todo: 这个的可视化太麻烦了，不太容易使用，准备重构

"""
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import ConnectionPatch
from pandas import DataFrame
from PIL import Image, ImageDraw
from pretty_tools.datastruct import np_ltrb_to_xywh
from pretty_tools.datastruct.multi_index_dict import mdict

from .draw_track import Visual_Track


class Visual_Track_MTMCT(Visual_Track):
    """
    保存了多个可视化工具的多相机可视化工具
    本身就是一个Visual_Track类
    """

    def __init__(self, outline=1):
        super().__init__(outline=outline)
        self.dict_visual: dict[str, Visual_Track] = {}
        self.__num_camera = 0
        self.list_visual: list[Visual_Track] = []

        # * 在计算的时候可能用到的数据
        self.dict_img: dict["str|int", Image.Image] = {}  # 保存了每个相机的图像
        self.dict_visual_image: dict["str|int", Image.Image] = {}  # 保存了每个相机的可视化图像
        self.dict_ltrb: dict["str|int", np.ndarray] = {}  # 保存了每个相机内目标的归一化锚框
        self.dict_merged_offset_ori_ltrb: dict[str, np.ndarray] = {}  # 保存了每个相机内带偏移量的目标的归一化锚框
        self.dict_ids: dict["str|int", np.ndarray] = {}  # 保存了每个相机内目标的id
        self.dict_gt: dict["str|int", np.ndarray] = {}  # 保存了每个相机内目标的gt
        self.mdict_similarity: mdict = mdict(2)  # 保存了相机之间的相似度矩阵
        self.background_image: Optional[Image.Image] = None  # 保存了可视化的背景图像
        self.mdict_pair: mdict[int, int, np.ndarray]  # 保存了相机之间的匹配关系
        self.pair_gt: DataFrame

    def __getitem__(self, name_camera):
        if name_camera not in self.dict_visual:
            self.__num_camera += 1
            new_visual = Visual_Track(outline=self.outline)
            self.dict_visual[name_camera] = new_visual
            self.list_visual.append(new_visual)
        return self.dict_visual[name_camera]

    def visual_batch(
        self,
        dict_img: Dict[str, Image.Image] = None,  # type: ignore
        dict_ltrb: Dict[str, np.ndarray] = None,  # type: ignore
        dict_gt: Dict[str, np.ndarray] = None,  # type: ignore
        dict_ids: Dict[str, np.ndarray] = None,  # type: ignore
        size: Tuple[int, int] = None,  # type: ignore
    ):
        if dict_img is not None:
            self.dict_img = dict_img
        if dict_ltrb is not None:
            self.dict_ltrb = dict_ltrb
        if dict_gt is not None:
            self.dict_gt = dict_gt
        if dict_ids is not None:
            self.dict_ids = dict_ids
        if size is None:
            size = (1920, 1080)

        self.dict_visual_image = {}
        for i, (name, img) in enumerate(self.dict_img.items()):
            ltrb = self.dict_ltrb[name]
            ids = self.dict_ids[name]
            # img = cv2.imread(str(camera.list_imgs[frame_index]))
            # img = cv2.resize(img, size) if (int(img.shape[1]), int(img.shape[0])) != size else img
            img = img.resize(size)
            visual_img = self[name](img, ltrb, ids)  #! 这里进行了可视化绘制
            if len(self.dict_gt) > 0:
                gt = self.dict_gt[name]
                visual_img = self[name].draw_gt(visual_img, gt)  #! 这里进行了可视化绘制

            self.dict_visual_image[name] = visual_img

        return self.dict_visual_image

    def visual_merge(
        self,
        size: Tuple[int, int] = None,  # type: ignore
    ):
        """
        类中内置的一个可视化函数

        dict_ltrb 建议使用归一化的锚框

        *注意: frame_index 从 0 开始
        todo 目前只能将所有图像都统一到同样的尺寸进行可视化
        todo 目前的可视化拼接只能向下拼接

        Image (h,w,c)
        cv2   (w,h,c)
        """
        if size is None:
            size = (1920, 1080)
        self.background_image = Image.new("RGB", (size[0], len(self.dict_img) * size[1]), (0, 0, 0))

        for i, (name, img) in enumerate(self.dict_visual_image.items()):
            img = img.resize(size)

            if len(self.dict_gt) > 0:
                gt = self.dict_gt[name]
                visual_img = self[name].draw_gt(visual_img, gt)  #! 这里进行了可视化绘制

            self.dict_visual_image[name] = visual_img
            self.background_image.paste(visual_img, (0, i * visual_img.size[1]))
        return self.background_image

    def draw_affinity(
        self,
        mdict_similarity: mdict,
        dict_ids: Dict[str, np.ndarray] = None,  # type: ignore
        dict_img: Dict[str, Image.Image] = None,  # type: ignore
        mdict_pair: mdict = None,  # type: ignore
        reid_shape: Tuple[int, int] = (128, 256),  # * (w,h)
        dpi=200,
    ):
        """
        #! 已被重构
        绘制亲合度矩阵
        pretty_tools >= 0.1.5 后，这个函数使用的 mdict_pair 是一个mdict类型的数据
        """
        z_t = 0.8
        if mdict_similarity is not None:
            self.mdict_similarity = mdict_similarity
        if dict_ids is not None:
            self.dict_ids = dict_ids
        if dict_img is not None:
            self.dict_img = dict_img
        if mdict_pair is not None:
            self.mdict_pair = mdict_pair
        ratio_hw = reid_shape[1] / reid_shape[0]
        assert self.mdict_similarity.dim == 2
        os.makedirs("tmp", exist_ok=True)
        max_m = sum([i.shape[1] for i in self.mdict_similarity.values()])
        max_n = max([i.shape[0] for i in self.mdict_similarity.values()])  # * 根据图像特征比例扩大垂直方向的尺寸

        dict_list_cut: Dict[str, List[Image.Image]] = {}
        for camera_name, img in self.dict_img.items():
            dict_list_cut[camera_name] = self.dict_visual[camera_name].cut_box(img, self.dict_ltrb[camera_name])
            dict_list_cut[camera_name] = [i.resize(reid_shape) for i in dict_list_cut[camera_name]]

        fig, axs = plt.subplots(nrows=1, ncols=len(self.mdict_similarity), figsize=(max_m, max_n), dpi=dpi)  # * 多出来的 1 是放图像的
        fig.subplots_adjust(wspace=0.5, hspace=0.5, top=z_t)
        for i, ((camera1, camera2), matrix) in enumerate(self.mdict_similarity.items()):
            if i != len(self.mdict_similarity):
                pic = sns.heatmap(matrix, annot=True, cmap="coolwarm", vmin=0, vmax=1, ax=axs[i], cbar=False)  # * 使用统一的颜色条
            else:
                pic = sns.heatmap(matrix, annot=True, cmap="coolwarm", vmin=0, vmax=1, ax=axs[i], cbar_ax=[axs[i]])  # * 使用统一的颜色条
            zw = dpi / reid_shape[0] / matrix.shape[1]  ## matrix.shape[1] # 是相似度矩阵水平方向的尺寸
            zh = dpi / reid_shape[1] / matrix.shape[0]  ## matrix.shape[0] # 是相似度矩阵垂直方向的尺寸
            for t, image in enumerate(dict_list_cut[camera2]):
                imagebox_w = OffsetImage(image, zoom=zw * z_t)
                imagebox_w.image.axes = pic
                pic.add_artist(AnnotationBbox(imagebox_w, (t + 0.5, 0), xybox=(t + 0.5, -0.3 * zw / zh), frameon=False))
            for t, image in enumerate(dict_list_cut[camera1]):
                imagebox_h = OffsetImage(image, zoom=zh * ratio_hw * z_t)  # 0.8 z_t
                imagebox_h.image.axes = pic
                pic.add_artist(AnnotationBbox(imagebox_h, (0, t + 0.5), xybox=(-1 * zh / zw, t + 0.5), frameon=False))
            if self.mdict_pair is not None:  # * 高亮匹配的框
                for pair in np.array(self.mdict_pair.loc[camera1, camera2]):
                    text = pic.texts[int(pair[0] * matrix.shape[1] + pair[1])]
                    text.set_size(14)
                    text.set_weight("bold")
                    text.set_style("italic")

            pic.set_xlabel(camera2, labelpad=0)
            pic.set_ylabel(camera1, labelpad=80 * zh / zw)
            pic.yaxis.tick_right()  # * y轴标签放在右边
        fig.colorbar(axs[-1].collections[0], ax=axs, location="right")  # 设置整个画布的颜色条
        plt.savefig("tmp/affinity.jpg")
        plt.close()
        pass

    def draw_match(
        self,
        dict_img: Dict[str, Image.Image] = None,  # type: ignore
        dict_ltrb: Dict[str, np.ndarray] = None,  # type: ignore
        dict_ids: Dict[str, np.ndarray] = None,  # type: ignore
        mdict_pair: mdict = None,  # type: ignore
        size: Tuple[int, int] = (1920, 1080),
        thresh_similarity: "float | List[float]" = 0.0,
    ):
        """
        pretty_tools >= 0.1.5, 使用的pair_infer类型为 mdict
        """
        if dict_img is not None:
            self.dict_img = dict_img
        if dict_ltrb is not None:
            self.dict_ltrb = dict_ltrb
        if dict_ids is not None:
            self.dict_ids = dict_ids
        if mdict_pair is not None:
            self.mdict_pair = mdict_pair
        if isinstance(thresh_similarity, float):
            thresh_similarity = [thresh_similarity]
        assert self.mdict_pair is not None
        list_name = list(self.dict_img.keys())

        num_camera = len(list_name)
        # * 检查ltrb是否是归一化的
        for i in self.dict_ltrb.values():
            assert (i < 2).all() and (-1 < i).all()

        for thresh in thresh_similarity:
            num_iter = math.comb(num_camera, 2)

            # 生成子图
            fig, axs = plt.subplots(nrows=num_iter, ncols=2, figsize=(10 * num_iter, 10 * 2), dpi=200)
            fig.subplots_adjust(wspace=0, hspace=0.2)

            iter_axs = iter(axs)
            iter_line_color = iter([tuple(i) for i in np.random.randint(0, 255, (num_iter, 3)).astype(int)])

            for i in range(0, num_camera - 1):
                for j in range(i + 1, num_camera):
                    name_a = list_name[i]
                    name_b = list_name[j]

                    link_index: np.ndarray = mdict_pair[i, j]  # type: ignore

                    # * ========================= 可视化方法二 =================================

                    ltrb_a = self.dict_ltrb[name_a]
                    ltrb_b = self.dict_ltrb[name_b]
                    _xywh_a: np.ndarray = np_ltrb_to_xywh(ltrb_a[link_index[:, 0]])
                    _xywh_b: np.ndarray = np_ltrb_to_xywh(ltrb_b[link_index[:, 1]])

                    ax_a, ax_b = next(iter_axs)
                    assert len(self.dict_visual_image) > 0, "需要事先可视化标注好的图像"
                    visual_image_a = self.dict_visual_image[name_a]
                    visual_image_b = self.dict_visual_image[name_b]
                    ax_a.imshow(visual_image_a)
                    ax_b.imshow(visual_image_b)
                    for bbox_a, bbox_b in zip(_xywh_a, _xywh_b):
                        xy_a = bbox_a[:2] * visual_image_a.size
                        xy_b = bbox_b[:2] * visual_image_b.size
                        con = ConnectionPatch(
                            xyA=xy_a,
                            xyB=xy_b,
                            coordsA="data",
                            coordsB="data",
                            axesA=ax_a,
                            axesB=ax_b,
                            arrowstyle="->",
                            color=tuple(np.random.rand(3)),
                            connectionstyle=f"arc3,rad={0.2*(random.random() -0.5)}",  # (-0.2,0.2) 的一个随机弧度
                        )
                        ax_b.add_artist(con)
                        ax_a.plot(*xy_a, "ro", markersize=4)
                        ax_b.plot(*xy_b, "ro", markersize=4)
                        pass

            plt.savefig(f"tmp/match_thresh_{thresh:0.2f}.jpg")
            # * ==========================================================
        pass
