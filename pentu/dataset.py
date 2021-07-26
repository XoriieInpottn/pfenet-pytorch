#!/usr/bin/env python3

"""
@author: xi
@since: 2021-07-19
"""

import collections
import os
import random
from typing import Iterable

import cv2 as cv
import numpy as np
from imgaug import SegmentationMapsOnImage
from imgaug import augmenters as iaa
from torch.utils.data import Dataset
from tqdm import tqdm

MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def encode_image(image: np.ndarray) -> np.ndarray:
    """Convert an image to float32 and CHW format.

    :param image: np.ndarray, dtype=uint8, shape=(h, w, c)
    :return: np.ndarray, dtype=float32, shape=(c, h, w)
    """
    image = image.astype(np.float32)
    image = (image - MEAN) / STD
    image = np.transpose(image, (2, 0, 1))
    return image


def decode_image(tensor: np.ndarray) -> np.ndarray:
    """Convert float tensor back to an image.

    :param tensor: np.ndarray, dtype=float32, shape=(c, h, w)
    :return: np.ndarray, dtype=uint8, shape=(h, w, c)
    """
    tensor = np.transpose(tensor, (1, 2, 0))
    tensor = tensor * STD + MEAN
    tensor = np.clip(tensor, 0, 255)
    return tensor.astype(np.uint8)


def encode_label(label: np.ndarray) -> np.ndarray:
    return (label / 255).astype(np.int64)


def decode_label(tensor: np.ndarray) -> np.ndarray:
    return (tensor * 255).astype(np.uint8)


class AugmenterWrapper(object):

    def __init__(self, augmenters: Iterable[iaa.Augmenter]):
        self._augmenter = iaa.Sequential(augmenters)

    def __call__(self, image, mask):
        seg_maps = SegmentationMapsOnImage(mask, shape=mask.shape)
        image, seg_maps = self._augmenter(
            image=image, segmentation_maps=seg_maps)
        mask = seg_maps.arr
        mask = mask.squeeze(2)
        return image, mask


IGNORE_CLASS = 300


class SegmentationDataset(Dataset):
    """Dataset for k-shot image segmentation.
    """

    def __init__(self,
                 path,
                 sub_class_list,
                 num_shots,
                 image_size,
                 is_train=False,
                 parse_class=False):
        self._num_shots = num_shots
        if not isinstance(image_size, (tuple, list)):
            image_size = (image_size, image_size)
        self._transform = AugmenterWrapper([
            iaa.Resize({'longer-side': (473, 500),
                        'shorter-side': 'keep-aspect-ratio'}, 'linear'),
            iaa.Fliplr(0.5),
            iaa.Rotate((-10, 10), cval=127),
            iaa.GaussianBlur((0.0, 0.1)),
            iaa.PadToAspectRatio(1.0, pad_cval=127, position='center-center'),
            iaa.CropToFixedSize(473, 473)
        ]) if is_train else AugmenterWrapper([
            iaa.Resize(
                {'longer-side': 473, 'shorter-side': 'keep-aspect-ratio'}, 'linear'),
            iaa.PadToAspectRatio(1.0, pad_cval=127, position='center-center')
        ])

        task_list = []
        for filename in os.listdir(path):
            task_list.append(filename)
        docs = []
        for task_i in task_list:
            image_path = os.path.join(path, task_i, 'image')
            label_path = os.path.join(path, task_i, 'mask')
            image_list = os.listdir(image_path)
            label_list = os.listdir(label_path)
            image_list.sort()
            label_list.sort()
            assert (len(image_list) == len(label_list))
            for j in range(len(image_list)):
                single_image_path = os.path.join(image_path, image_list[j])
                single_mask_path = os.path.join(label_path, label_list[j])
                doc = {'image': single_image_path,
                       'label': single_mask_path, 'class': task_i}
                docs.append(doc)

        self._doc_list = []
        self._sub_class_dict = collections.defaultdict(list)
        for doc in tqdm(docs, leave=False):
            c = doc['class']
            self._doc_list.append((doc, c))
            self._sub_class_dict[c].append(doc)

    def __len__(self):
        return len(self._doc_list)

    def __getitem__(self, i):
        doc, class_chosen = self._doc_list[i]
        image = cv.imread(doc['image'], cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        label = cv.imread(doc['label'], cv.IMREAD_GRAYSCALE)
        if callable(self._transform):
            image, label = self._transform(image, label)
        query_doc = {
            'image': encode_image(image),
            'label': encode_label(label),
            'class': class_chosen
        }

        supp_image_list = []
        supp_label_list = []
        docs_chosen = random.sample(
            self._sub_class_dict[class_chosen], self._num_shots)
        for doc in docs_chosen:
            image = cv.imread(doc['image'], cv.IMREAD_COLOR)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            label = cv.imread(doc['label'], cv.IMREAD_GRAYSCALE)

            if callable(self._transform):
                image, label = self._transform(image, label)
            supp_image_list.append(encode_image(image))
            supp_label_list.append(encode_label(label))
        supp_doc = {
            'image': np.stack(supp_image_list),  # (k, c, h, w)
            'label': np.stack(supp_label_list)  # (k, h, w)
        }
        return supp_doc, query_doc


def test():
    ds = SegmentationDataset(
        '/edgeai/shared/PEN_TU_DATA/SMD/train',
        ['ele_up'],
        num_shots=5,
        image_size=(473, 473),
        is_train=True
    )
    supp_doc, query_doc = ds[10]
    image = supp_doc['image'][2]
    label = supp_doc['label'][2]
    image = decode_image(image)
    label = decode_label(label)
    cv.imwrite('image.jpg', image)
    cv.imwrite('label.png', label)
    print(image.mean(), label.mean())
    exit()

    from torch.utils.data import DataLoader
    loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=False,
        num_workers=40,
        pin_memory=True
    )
    print(len(loader))
    loop = tqdm(loader, dynamic_ncols=True, leave=False)
    for supp_doc, query_doc in loop:
        label = query_doc['label']
        uniou = (label[3] == 1).sum()
        print(uniou)
        exit()
    return 0


if __name__ == '__main__':
    raise SystemExit(test())
