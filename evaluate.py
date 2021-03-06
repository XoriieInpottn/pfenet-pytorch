#!/usr/bin/env python3

"""
@author: Haoyu, Guangyi
@since: 2021-07-21
"""

import collections
from typing import List

import cv2 as cv
import numpy as np


class IouMeter(object):

    def __init__(self, ignore_class: int, eps=1e-10):
        self._ignore_class = ignore_class
        self._eps = eps

        self._inter_fg = 0
        self._union_fg = 0
        self._inter_bg = 0
        self._union_bg = 0
        self._pred = 0
        self._true = 0
        self._class_inter = collections.defaultdict(int)
        self._class_union = collections.defaultdict(int)

    def update(self,
               output: np.ndarray,
               target: np.ndarray,
               class_list: List[int]):
        """Update the meter's state by a batch of result.

        :param output: dtype=int64, shape=(n, h, w)
        :param target: dtype=int64, shape=(n, h, w)
        :param class_list: list of classes
        """
        assert len(output) == len(target) == len(class_list)
        for output_i, target_i, class_i in zip(output, target, class_list):
            output_i = output_i.copy()  # if you don't copy, you will corrupt the original input
            output_i[np.where(target_i == self._ignore_class)] = self._ignore_class

            inter = ((output_i == 1) & (target_i == 1)).sum()
            union = ((output_i == 1) | (target_i == 1)).sum()
            self._class_inter[class_i] += inter
            self._class_union[class_i] += union
            self._inter_fg += inter
            self._union_fg += union

            pred = (output_i == 1).sum()
            true = (target_i == 1).sum()
            self._pred += pred
            self._true += true

            inter_bg = ((output_i == 0) & (target_i == 0)).sum()
            union_bg = ((output_i == 0) | (target_i == 0)).sum()
            self._inter_bg += inter_bg
            self._union_bg += union_bg

    def m_iou(self):
        iou_list = []
        for clazz in self._class_inter:
            inter = self._class_inter[clazz]
            union = self._class_union[clazz]
            iou_list.append(inter / (union + self._eps))
        return np.mean(iou_list)

    def fb_iou(self):
        iou_fg = self._inter_fg / (self._union_fg + self._eps)
        iou_bg = self._inter_bg / (self._union_bg + self._eps)
        return (iou_fg + iou_bg) * 0.5

    def precision(self):
        return self._inter_fg / (self._pred + self._eps)

    def recall(self):
        return self._inter_fg / (self._true + self._eps)


def draw_mask(image, mask):
    mask = mask.astype(np.uint8)
    mask = np.clip(mask, 0, 1) * 255
    mask = np.stack([mask, mask, np.zeros_like(mask)], 2)
    image = cv.addWeighted(image, 0.5, mask, 0.5, 0)
    return image
