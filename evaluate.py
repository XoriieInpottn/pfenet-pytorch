#!/usr/bin/env python3

"""
@author: Haoyu
@since: 2021-07-21
"""

import torch
import dataset
import pfenet
import numpy as np


class Evaluate(object) :
    '''
    test model 
    '''
    def __init__(self):
        self._miou_dict={}
        self._fb_iou=[]
    

    def update(self,pred,gt_mask,class_num):
        '''
        Args:
            pred(numpy):(n,h,w)
            gt_mask(numpy):(n,h,w)
            class_num(list):(n)
        '''
        for i in range(pred.shape[0]):
            
            pred[i][np.where(gt_mask[i] == 255)] = 255
         
            intersection = ((pred[i] == 1) & (gt_mask[i] == 1)).sum((0, 1))
            union = ((pred[i] == 1) | (gt_mask[i] == 1)).sum((0, 1))
    
            iou = float(intersection) / float(union)
            print(iou)

            self._fb_iou.append(iou)
            if class_num[i] in self._miou_dict:
                self._miou_dict[class_num].append(iou)
            else: 
                self._miou_dict[class_num]=[iou]

    def miou(self):
        miou_list=[]
        for key,value in self._miou_dict.items():
            miou_list.append(np.mean(value))
        return np.mean(miou_list)
    def fb_iou(self):
        
        return np.mean(self._fb_iou)


