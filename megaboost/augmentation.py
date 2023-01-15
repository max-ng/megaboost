# code in this file is adpated from
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
import logging
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
from audioop import bias

logger = logging.getLogger(__name__)

PARAMETER_MAX = 10
RESAMPLE_MODE = None


def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v, max_v, **kwarg):
    if v == 0:
        return img
    v = _float_parameter(v, max_v)
    v = int(v * min(img.size))
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def CutoutConst(img, v, max_v, **kwarg):
    v = _int_parameter(v, max_v)
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias, **kwarg):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, **kwarg):
    v = _float_parameter(v, max_v)
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, **kwarg):
    v = _float_parameter(v, max_v)
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0), RESAMPLE_MODE)


def ShearY(img, v, max_v, **kwarg):
    v = _float_parameter(v, max_v)
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0), RESAMPLE_MODE)


def Solarize(img, v, max_v, **kwarg):
    v = _int_parameter(v, max_v)
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, threshold=128, **kwarg):
    v = _int_parameter(v, max_v)
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, **kwarg):
    v = _float_parameter(v, max_v)
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0), RESAMPLE_MODE)


def TranslateY(img, v, max_v, **kwarg):
    v = _float_parameter(v, max_v)
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v), RESAMPLE_MODE)


def TranslateXConst(img, v, max_v, **kwarg):
    v = _float_parameter(v, max_v)
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0), RESAMPLE_MODE)


def TranslateYConst(img, v, max_v, **kwarg):
    v = _float_parameter(v, max_v)
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v), RESAMPLE_MODE)


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def rand_augment_pool():
    augs = [(AutoContrast, None, None),
            (Brightness, 1.8, 0.1),
            (Color, 1.8, 0.1),
            (Contrast, 1.8, 0.1),
            (CutoutConst, 40, None),
            (Equalize, None, None),
            (Invert, None, None),
            (Posterize, 4, 0),
            (Rotate, 30, None),
            (Sharpness, 1.8, 0.1),
            (ShearX, 0.3, None),
            (ShearY, 0.3, None),
            (Solarize, 256, None),
            (TranslateXConst, 100, None),
            (TranslateYConst, 100, None),
            ]
    return augs


class RandAugment(object):
    def __init__(self, n, m, resample_mode=PIL.Image.BILINEAR):
        assert n >= 1
        assert m >= 1
        global RESAMPLE_MODE
        RESAMPLE_MODE = resample_mode
        self.n = n
        self.m = m
        self.augment_pool = rand_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            prob = np.random.uniform(0.2, 0.8)
            if random.random() + prob >= 1:
                img = op(img, v=self.m, max_v=max_v, bias=bias)
        return img

class CutOutAug(object):
    def __init__(self, n, size, resample_mode=PIL.Image.BILINEAR):
        assert n >= 1
        global RESAMPLE_MODE
        RESAMPLE_MODE = resample_mode
        self.n = n
        self.size =size
        self.augment_pool = [(CutoutConst, 40, None)]

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            prob = np.random.uniform(0.2, 0.8)
            if random.random() + prob >= 1:
                img = op(img, v=self.size, max_v=self.size, bias=bias)
        return img
