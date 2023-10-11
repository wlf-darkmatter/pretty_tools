import cv2
import numpy as np


def convert_rgb_to_bgr(img: np.ndarray):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def convert_bgr_to_rgb(img: np.ndarray):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
