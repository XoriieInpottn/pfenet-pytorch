#!/usr/bin/env python3

"""
@author: Haoyu, Guangyi
@since: 2021-07-21
"""

import collections
from typing import List

import numpy as np


class ClassIouMeter(object):

    def __init__(self, ignore_class: int):
        self._ignore_class = ignore_class
        self._m_iou_dict = collections.defaultdict(list)
        self._fb_iou = []

    def update(self,
               pred: np.ndarray,
               target: np.ndarray,
               class_list: List[int]):
        """Update the meter's state by a batch of result.

        :param pred: dtype=int64, shape=(n, h, w)
        :param target: dtype=int64, shape=(n, h, w)
        :param class_list: list of classes
        """
        assert len(pred) == len(target) == len(class_list)
        for pred_i, target_i, class_i in zip(pred, target, class_list):
            pred_i[np.where(target_i == self._ignore_class)] = self._ignore_class

            intersection = ((pred_i == 1) & (target_i == 1)).sum()
            union = ((pred_i == 1) | (target_i == 1)).sum()
            iou = float(intersection) / float(union)

            self._fb_iou.append(iou)
            self._m_iou_dict[class_i].append(iou)

    def m_iou(self):
        m_iou_list = []
        for iou_list in self._m_iou_dict.values():
            m_iou_list.append(np.mean(iou_list))
        return np.mean(m_iou_list)

    def fb_iou(self):
        return np.mean(self._fb_iou)
