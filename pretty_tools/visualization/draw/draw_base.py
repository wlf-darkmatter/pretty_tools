import copy
from functools import partial

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ...resources import path_font_arial
from ..core import log_visualization


def draw_bboxes(img, bboxes_ltrb: np.ndarray, colors=None, infos=None, outline=2, size_font=24, mask: float = 0) -> Image.Image:
    """
    bboxes的格式一定是左上宽高
    """
    if colors is None:
        colors = (255, 255, 255)
    img = copy.deepcopy(img)
    bboxes_ltrb = copy.deepcopy(bboxes_ltrb)
    #! 一定要是np.float32或者直接用int
    if bboxes_ltrb.dtype == np.float64 or bboxes_ltrb.dtype == np.float128:
        bboxes_ltrb = bboxes_ltrb.astype(np.float32)
    bboxes_ltrb = bboxes_ltrb.astype(np.float32)
    if infos is not None:
        assert len(bboxes_ltrb) == len(infos)
    if isinstance(img, np.ndarray):
        image = Image.fromarray(img)
    else:
        image = img

    if bboxes_ltrb.ndim == 1:
        bboxes_ltrb = bboxes_ltrb[np.newaxis, :]

    if type(colors) == tuple:
        colors = len(bboxes_ltrb) * [colors]

    # * 绘制具有透明度的图片

    if mask != 0:
        mask = int(255 * mask)
        _image = image.convert("RGBA")
        image = Image.new("RGBA", img.size, (0, 0, 0, 0))

        colors: List[tuple] = [(*color, 255) for color in colors]  # type:ignore
        #! 如果有透明度，则自上而下地绘制，下面的覆盖上面的
        sort_index = np.argsort(bboxes_ltrb[:, 3])
        colors = [colors[i] for i in sort_index]
        infos = [infos[i] for i in sort_index] if infos is not None else infos
        bboxes_ltrb = bboxes_ltrb[sort_index]
        # * 绘制mask
        draw = ImageDraw.Draw(image, mode="RGBA")
        for box, color in zip(bboxes_ltrb, colors):
            draw.rectangle(box, width=outline, outline=color, fill=color[:3] + (mask,))
        image = Image.alpha_composite(_image, image)
        image = image.convert("RGB")

    draw = ImageDraw.Draw(image, mode="RGB")
    # * 绘制边框 不能和mask一起绘制，会出现边框被mask覆盖的问题
    for box, color in zip(bboxes_ltrb, colors):
        draw.rectangle(box, width=outline, outline=color)

    if infos is not None:
        font = ImageFont.truetype(str(path_font_arial), size_font)
        for box, info in zip(bboxes_ltrb, infos):
            draw.text(box[0:2], info, font=font)
    return image


def draw_title(img, title: str, size_scale: float = 1.0) -> Image.Image:
    if type(img) == np.array:
        image = Image.fromarray(img)
    else:
        image = img
    real_scale = image.width / 1920 * size_scale
    size_font = int(24 * real_scale)
    font = ImageFont.truetype(str(path_font_arial), size_font)
    draw = ImageDraw.Draw(image)

    line_title = title.split("\n")
    row = 0
    for i, line in enumerate(line_title):
        text_width, text_height = draw.textsize(line, font=font)
        row += text_height
        draw.text((size_font - real_scale, row - real_scale), line, align="center", font=font, fill=(0, 0, 0))  # * 添加描边
        draw.text((size_font + real_scale, row - real_scale), line, align="center", font=font, fill=(0, 0, 0))  # * 添加描边
        draw.text((size_font - real_scale, row + real_scale), line, align="center", font=font, fill=(0, 0, 0))  # * 添加描边
        draw.text((size_font + real_scale, row + real_scale), line, align="center", font=font, fill=(0, 0, 0))  # * 添加描边
        draw.text((size_font, row), line, align="center", font=font, fill=(255, 255, 255))

    return image


def draw_gt_bboxes(img, bboxes_ltrb: np.ndarray, infos=None, outline=2, size_font=24, mask: float = 0) -> Image.Image:
    """
    bboxes的格式一定是左上宽高

    绘制 gt bbox，绘制的时候用黑白双线框
    """
    img = copy.deepcopy(img)
    bboxes_ltrb = copy.deepcopy(bboxes_ltrb)
    #! 一定要是np.float32或者直接用int
    if bboxes_ltrb.dtype == np.float64 or bboxes_ltrb.dtype == np.float128:
        bboxes_ltrb = bboxes_ltrb.astype(np.float32)
    bboxes_ltrb = bboxes_ltrb.astype(np.float32)
    if infos is not None:
        assert len(bboxes_ltrb) == len(infos)
    if type(img) == np.array:
        image = Image.fromarray(img)
    else:
        image = img
    draw = ImageDraw.Draw(image)
    if bboxes_ltrb.ndim == 1:
        bboxes_ltrb = bboxes_ltrb[np.newaxis, :]

    # * 绘图

    for box in bboxes_ltrb:
        draw.rectangle(box, outline=(255, 255, 255), width=outline + 1)
        draw.rectangle(box, outline=(0, 0, 0), width=outline)

    if infos is not None:
        font = ImageFont.truetype(str(path_font_arial), size_font)
        for box, info in zip(bboxes_ltrb, infos):
            draw.text(box[[2, 1]], info, font=font)  # * GT 信息绘制在右上角

    return image
