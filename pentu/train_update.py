#!/usr/bin/env python3
'''
Author: Haoyu
Date: 2021-07-28 16:08:41
'''

import argparse
import os

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import functional as F

import dataset
import pfenet_memory
import utils
from evaluate import IouMeter
from update import Memory_network

writer = SummaryWriter('run')


class Trainer(object):

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu', type=str, default='2',
                            help='Which GPU to use.')
        parser.add_argument('--batch-size', type=int,
                            default=8, help='Batch size.')
        parser.add_argument('--num-epochs', type=int,
                            default=100, help='The number of epochs to train.')
        parser.add_argument('--max-lr', type=float, default=1e-3,
                            help='The maximum value of learning rate.')
        parser.add_argument('--weight-decay', type=float,
                            default=0.3, help='The weight decay value.')
        parser.add_argument('--optimizer', default='AdamW',
                            help='Name of the optimizer to use.')

        parser.add_argument('--num-shots', type=int, default=5)
        parser.add_argument('--image-size', type=int, default=473)
        parser.add_argument('--output-dir', default='output')
        self._args = parser.parse_args()
        os.environ['CUDA_VISIBLE_DEVICES'] = self._args.gpu
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._create_dataset()
        self._create_model()
        self._create_optimizer()

    def _create_dataset(self):
        train_dataset = dataset.SegmentationDataset(
            '/edgeai/shared/PEN_TU_DATA/SMD/train',
            ['ele_up', 'ele_miss'],
            num_shots=self._args.num_shots,
            image_size=self._args.image_size,
            is_train=True
        )

        def worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        self._train_loader = DataLoader(
            train_dataset,
            batch_size=self._args.batch_size,
            shuffle=False,
            num_workers=10,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )
        test_dataset = dataset.SegmentationDataset(
            '/edgeai/shared/PEN_TU_DATA/SMD/test',
            ['ele_up'],
            num_shots=self._args.num_shots,
            image_size=self._args.image_size,
        )
        self._test_loader = DataLoader(
            test_dataset,
            batch_size=self._args.batch_size,
            num_workers=10,
            pin_memory=True
        )

    def _create_model(self):
        self._model = pfenet_memory.PFENet(
            *pfenet_memory.get_mobilenet_v2_layers())
        self._model = self._model.to(self._device)
        self._model = torch.load('saved_model.pkl')
        #for para in self._model.parameters():
        #     para.requires_grad  = False
        self._model.memory_network = Memory_network(256, 10)
        for p in self._model.memory_network.fc_read.parameters():
            p.requires_grad = True
        for p in self._model.memory_network.fc_write.parameters():
            p.requires_grad = True
        # "requires_grad" of of the backbone parameters are set to False
        self._parameters = [
            p for p in self._model.memory_network.parameters()]# if p.requires_grad]
        self._loss = pfenet_memory.Loss()

    def _create_optimizer(self):
        if self._args.optimizer == 'MomentumSGD':
            self._optimizer = optim.SGD(
                self._parameters,
                lr=self._args.max_lr,
                weight_decay=self._args.weight_decay,
                momentum=0.9
            )
        else:
            optimizer_class = getattr(optim, self._args.optimizer)
            self._optimizer = optimizer_class(
                self._parameters,
                lr=self._args.max_lr,
                weight_decay=self._args.weight_decay,
            )
        num_loops = self._args.num_epochs * len(self._train_loader)
        self._scheduler = utils.CosineWarmUpAnnealingLR(
            self._optimizer, num_loops)

    def train(self):
        loss_g = 0.0
        for epoch in range(self._args.num_epochs):
            loop = tqdm(self._train_loader, dynamic_ncols=True, leave=False)
            
            # loop.write(
            #     f'[{epoch + 1}/{self._args.num_epochs}] '

            #     f'mIOU={m_iou:.02%} '
            #     f'fbIOU={fb_iou:.02%} '
            #     f'recall={recall:.02%}'
            #     f'precision={precision:.02%}'
            # )

            self._model.train()
            for supp_doc, query_doc in loop:
                loss, lr, pred = self._train_step(
                    supp_doc['image'],
                    supp_doc['label'],
                    query_doc['image'],
                    query_doc['label']
                )
                loss = float(loss.numpy())
                loss_g = 0.9 * loss_g + 0.1 * loss
                loop.set_description(
                    f'[{epoch + 1}/{self._args.num_epochs}] L={loss_g:.06f} lr={lr:.01e}', False)

            writer.add_scalar('loss', loss_g, epoch)
            self._model.eval()
            m_iou, fb_iou, recall, precision = self._evaluate()
            loop.write(
                f'[{epoch + 1}/{self._args.num_epochs}] '
                f'L={loss_g:.06f} '
                f'mIOU={m_iou:.02%} '
                f'fbIOU={fb_iou:.02%} '
                f'recall={recall:.02%}'
                f'precision={precision:.02%}'
            )

    def _evaluate(self):
        meter = IouMeter(dataset.IGNORE_CLASS)
        loop = tqdm(self._test_loader, dynamic_ncols=True, leave=False)
        for supp_doc, query_doc in loop:
            output = self._predict_step(
                supp_doc['image'],
                supp_doc['label'],
                query_doc['image']
            )
            target = query_doc['label']  # (n, h, w)

            output = output.numpy()
            target = target.numpy()
            class_list = [clazz for clazz in query_doc['class']]

            meter.update(output, target, class_list)
            m_iou = meter.m_iou()
            writer.add_image('query label', target[0], dataformats='HW')
            writer.add_image('pred', output[0], dataformats='HW')
            writer.add_image('query image', query_doc['image'][0])

        return meter.m_iou(), meter.fb_iou(), meter.recall(), meter.precision()

    def _train_step(self, sx, sy, qx, qy):
        sx = sx.to(self._device)
        sy = sy.to(self._device)
        qx = qx.to(self._device)
        qy = qy.to(self._device)
        # sy[torch.where(torch.eq(sy, dataset.IGNORE_CLASS))] = 0  # clear the "ignore" class
        # qy[torch.where(torch.eq(qy, dataset.IGNORE_CLASS))] = 0  # clear the "ignore" class
        output, aux_list = self._model(sx, sy, qx)
        loss = self._loss(output, qy, aux_list)
        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()
        self._scheduler.step()
        return loss.detach().cpu(), self._scheduler.get_last_lr()[0], output

    def _predict_step(self, sx, sy, qx):
        sx = sx.to(self._device)
        sy = sy.to(self._device)
        qx = qx.to(self._device)
        # sy[torch.where(torch.eq(sy, dataset.IGNORE_CLASS))] = 0  # clear the "ignore" class
        output, _ = self._model(sx, sy, qx)  # (n, num_classes, ?, ?)
        output = pfenet_memory.resize(
            output, (self._args.image_size, self._args.image_size))
        qy_ = torch.argmax(output, 1)
        return qy_.detach().cpu()


if __name__ == '__main__':
    raise SystemExit(Trainer().train())
