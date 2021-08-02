#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2021-07-16
"""
from typing import List

import torch
from torch import nn
from torch.nn import functional as F


def resize(feat, size):
    return F.interpolate(feat, size=size, mode='bilinear', align_corners=True)


class PFENet(nn.Module):

    def __init__(self,
                 backbone_layers: List[nn.Module],
                 feat_size=256,
                 num_classes=2,
                 ppm_scales=(60, 30, 15, 8)):
        super(PFENet, self).__init__()
        self._ppm_scales = ppm_scales

        assert len(backbone_layers) >= 4
        self.backbone = nn.ModuleList(backbone_layers)
        for layer in self.backbone:
            layer.eval()
            for p in layer.parameters():
                p.requires_grad = False

        # detect the shape of every output
        dummy = torch.FloatTensor(1, 3, 100, 100)
        size_list = []
        for layer in self.backbone:
            dummy = layer(dummy)
            size_list.append(dummy.shape[1])
        backbone_feat_size = size_list[-2] + size_list[-3]

        self.down_query = nn.Sequential(
            nn.Conv2d(backbone_feat_size, feat_size, kernel_size=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(backbone_feat_size, feat_size, kernel_size=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.init_merge = nn.ModuleList()
        self.alpha_conv = nn.ModuleList()
        self.beta_conv = nn.ModuleList()
        self.inner_cls = nn.ModuleList()
        for i in range(len(self._ppm_scales)):
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(feat_size * 2 + 1, feat_size, kernel_size=(1, 1), bias=False),
                nn.ReLU(inplace=True)
            ))
            if i > 0:
                # alpha_conv is used to fuse the feature from the last scale
                # so at the first scale, there is nothing to conv
                self.alpha_conv.append(nn.Sequential(
                    nn.Conv2d(feat_size * 2, feat_size, kernel_size=(1, 1), bias=False),
                    nn.ReLU(inplace=True)
                ))
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(feat_size, feat_size, kernel_size=(3, 3), padding=(1, 1), bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(feat_size, feat_size, kernel_size=(3, 3), padding=(1, 1), bias=False),
                nn.ReLU(inplace=True)
            ))
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(feat_size, feat_size, kernel_size=(3, 3), padding=(1, 1), bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(feat_size, num_classes, kernel_size=(1, 1))
            ))

        self.res1 = nn.Sequential(
            nn.Conv2d(feat_size * len(self._ppm_scales), feat_size, kernel_size=(1, 1), bias=False),
            nn.ReLU(inplace=True)
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(feat_size, feat_size, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_size, feat_size, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True)
        )
        self.cls = nn.Sequential(
            nn.Conv2d(feat_size, feat_size, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(feat_size, num_classes, kernel_size=(1, 1))
        )

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            if module in set(self.backbone):
                continue
            module.train(mode)
        return self

    def forward(self, sx, sy, qx):
        """PFENet

        :param sx: dtype=float32, shape=(n, k, c, h, w)
        :param sy: dtype=int64, shape=(n, k, h, w)
        :param qx: dtype=float32, shape=(n, c, h, w)
        :return: out: dtype=float32, shape=(n, num_classes, output_h, output_w)
                 out_aux: dtype=float32, shape=(num_scales, n, num_classes, output_h, output_w)
        """
        # get the deep feature of the query sample
        # query_feat_c is used for generate the prior mask
        # query_feat is used for fusion and prediction
        with torch.no_grad():
            query_feat_list = []
            query_feat = qx
            for layer in self.backbone:
                query_feat = layer(query_feat)
                query_feat_list.append(query_feat)
            query_feat_a = query_feat_list[-3]
            query_feat_b = query_feat_list[-2]
            query_feat_c = query_feat_list[-1]

            query_feat_a_ = resize(query_feat_a, (query_feat_b.shape[2], query_feat_b.shape[3]))
            query_feat = torch.cat([query_feat_a_, query_feat_b], 1)

        query_feat = self.down_query(query_feat)  # (n, d, ?, ?)

        # get the deep feature of the support sample
        # supp_feat_c is used for generate the prior mask
        # supp_feat is used for fusion and prediction
        # note that supp_feat_c and supp_feat have different shape
        sx_flat = sx.view((-1, *sx.shape[2:]))  # (n, k, c, h, w) -> (nk, c, h, w)
        sy_flat = sy.float().view((-1, 1, *sy.shape[2:]))  # (n, k, h, w) -> (nk, 1, h, w)
        with torch.no_grad():
            supp_feat_list = []
            supp_feat = sx_flat
            for layer in self.backbone:
                supp_feat = layer(supp_feat)
                supp_feat_list.append(supp_feat)
            supp_feat_a = supp_feat_list[-3]
            supp_feat_b = supp_feat_list[-2]
            supp_feat_c = supp_feat_list[-1]
            supp_mask_c = resize(sy_flat, (supp_feat_c.shape[2], supp_feat_c.shape[3]))

            supp_feat_a_ = resize(supp_feat_a, (supp_feat_b.shape[2], supp_feat_b.shape[3]))
            supp_feat = torch.cat([supp_feat_a_, supp_feat_b], 1)
            supp_mask = resize(sy_flat, (supp_feat.shape[2], supp_feat.shape[3]))

        supp_feat = self.down_supp(supp_feat)  # (nk, d, ?, ?)

        supp_feat = supp_feat.reshape((sx.shape[0], -1, *supp_feat.shape[1:]))  # (n, k, d, ?, ?)
        supp_mask = supp_mask.reshape((sx.shape[0], -1, *supp_mask.shape[1:]))  # (n, k, 1, ?, ?)
        supp_feat_c = supp_feat_c.reshape((sx.shape[0], -1, *supp_feat_c.shape[1:]))  # (n, k, ?, ?, ?)
        supp_mask_c = supp_mask_c.reshape((sx.shape[0], -1, *supp_mask_c.shape[1:]))  # (n, k, 1, ?, ?)

        # compute prior
        prior = self._make_prior(supp_feat_c * supp_mask_c, query_feat_c)

        # compute prototype
        supp_feat = self._weighted_gap(supp_feat, supp_mask)  # (n, k, d, 1, 1)
        supp_feat = supp_feat.mean(1)  # (n, d, 1, 1)

        # compute pyramid features
        pyramid_feat_list, aux_list = self._pyramid(supp_feat, query_feat, prior)
        feat_size = (query_feat.size(2), query_feat.size(3))
        query_feat = torch.cat([resize(pyramid_feat, feat_size) for pyramid_feat in pyramid_feat_list], 1)

        # compute output
        query_feat = self.res1(query_feat)
        query_feat = self.res2(query_feat) + query_feat
        output = self.cls(query_feat)

        return output, aux_list

    @staticmethod
    def _weighted_gap(supp_feat, mask, eps=1e-4):
        # supp_feat: (n, k, d, h, w)
        # mask: (n, k, 1, h, w)
        weight = mask.mean((3, 4), keepdims=True) + eps
        return (supp_feat * mask).mean((3, 4), keepdims=True) / weight

    @staticmethod
    def _make_prior(supp_feat, query_feat):
        # supp_feat: (n, k, c, h, w)
        # query_feat: (n, c, h, w)
        n, k, c, h, w = supp_feat.shape
        query_feat = F.normalize(query_feat, 2, 1)
        supp_feat = F.normalize(supp_feat, 2, 2)
        query_feat = query_feat.view((n, c, -1))  # (n, c, hw)
        supp_feat = supp_feat.view(n, k, c, -1)  # (n, k, c, hw)
        sim = torch.einsum('nca,nkcb->nkab', query_feat, supp_feat).max(3)[0]  # (n, k, hw)
        sim_min = sim.min(2, keepdim=True)[0]  # (n, k, 1)
        sim_max = sim.max(2, keepdim=True)[0]  # (n, k, 1)
        prior = (sim - sim_min) / (sim_max - sim_min + 1e-10)
        prior = prior.view((n, k, h, w))  # (n, k, h, w)
        prior = prior.mean(1, keepdim=True)  # (n, 1, h, w)
        return prior

    def _pyramid(self, supp_feat, query_feat, prior):
        # supp_feat: (n, d, 1, 1)
        # query_feat: (n, d, h, w)
        pyramid_feat_list = []
        aux_list = []
        for idx, tmp_bin in enumerate(self._ppm_scales):
            if isinstance(tmp_bin, (tuple, list)):
                bin_h, bin_w = tmp_bin
            else:
                bin_h, bin_w = tmp_bin, tmp_bin
            if bin_h <= 1.0:
                bin_h = (int(query_feat.size(2) * bin_h), int(query_feat.size(3) * bin_h))
            if bin_w < 1.0:
                bin_w = (int(query_feat.size(2) * bin_w), int(query_feat.size(3) * bin_w))

            query_feat_bin = F.adaptive_avg_pool2d(query_feat, (bin_h, bin_w))  # (n, d, bin_h, bin_w)
            supp_feat_bin = supp_feat.expand(-1, -1, bin_h, bin_w)  # (n, d, bin_h, bin_w)
            prior_bin = resize(prior, (bin_h, bin_w))  # (n, 1, bin_h, bin_w)
            merge_feat_bin = torch.cat([query_feat_bin, supp_feat_bin, prior_bin], 1)  # (n, 2d + 1, bin_h, bin_w)
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)  # (n, d, bin_h, bin_w)

            if idx > 0:
                pre_feat_bin = pyramid_feat_list[idx - 1]
                pre_feat_bin = resize(pre_feat_bin, (bin_h, bin_w))
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx - 1](rec_feat_bin) + merge_feat_bin  # (n, d, bin_h, bin_w)

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin  # (n, d, bin_h, bin_w)
            pyramid_feat_list.append(merge_feat_bin)

            if self.training:
                inner_out_bin = self.inner_cls[idx](merge_feat_bin)  # (n, num_class, bin_h, bin_w)
                aux_list.append(inner_out_bin)

        return pyramid_feat_list, aux_list


class CrossEntropyLoss(nn.Module):

    def __init__(self, eps=1e-10):
        super(CrossEntropyLoss, self).__init__()
        self._eps = eps

    def forward(self, output, target, aux_list=None):
        """Cross-entropy loss.

        :param output: dtype=float32, shape=(n, c, ?, ?)
        :param target: dtype=int64, shape=(n, h, w)
        :param aux_list: list of dtype=float32, shape=(n, c, ?, ?)
        :return: loss
        """
        size = (target.size(1), target.size(2))

        target = F.one_hot(target, output.size(1)).float()  # (n, h, w, c)
        target = target.permute((0, 3, 1, 2))  # (n, c, h, w)
        output = resize(output, size)
        output = F.softmax(output, 1)
        loss = -target * torch.log(output + self._eps) - (1.0 - target) * torch.log(1.0 - output + self._eps)
        loss = loss.sum(1).mean()

        if aux_list:
            target = target.unsqueeze(0)  # (1, n, c, h, w)
            aux = torch.stack([resize(aux, size) for aux in aux_list])  # (?, n, c, h, w)
            aux = F.softmax(aux, 2)
            loss_aux = -target * torch.log(aux + self._eps) - (1.0 - target) * torch.log(1.0 - aux + self._eps)
            loss_aux = loss_aux.sum(2).mean(0).mean()
            loss = loss + loss_aux

        return loss


class FocalLoss(nn.Module):

    def __init__(self, eps=1e-10, gamma=3):
        super(FocalLoss, self).__init__()
        self._eps = eps
        self._gamma = gamma

    def forward(self, output, target, aux_list=None):
        """Focal loss.

        :param output: dtype=float32, shape=(n, c, ?, ?)
        :param target: dtype=int64, shape=(n, h, w)
        :param aux_list: list of dtype=float32, shape=(n, c, ?, ?)
        :return: loss
        """
        size = (target.size(1), target.size(2))

        target = F.one_hot(target, output.size(1)).float()  # (n, h, w, c)
        target = target.permute((0, 3, 1, 2))  # (n, c, h, w)
        output = resize(output, size)
        output = F.softmax(output, 1)
        p = (target * output).sum(1)
        loss = -torch.pow(1.0 - p, self._gamma) * torch.log(p + self._eps)
        loss = loss.sum((1, 2)).mean()

        if aux_list:
            target = target.unsqueeze(0)  # (1, n, c, h, w)
            aux = torch.stack([resize(aux, size) for aux in aux_list])  # (?, n, c, h, w)
            aux = F.softmax(aux, 2)

            p = (target * aux).sum(2)
            loss_aux = -torch.pow(1.0 - p, self._gamma) * torch.log(p + self._eps)
            loss_aux = loss_aux.sum((2, 3)).mean()
            loss = loss + loss_aux

        return loss


def get_vgg16_layers(pretrained=True):
    from torchvision.models import vgg
    net = vgg.vgg16_bn(pretrained=pretrained)
    layer0_idx = range(0, 7)
    layer1_idx = range(7, 14)
    layer2_idx = range(14, 24)
    layer3_idx = range(24, 34)
    layer4_idx = range(34, 43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [net.features[idx]]
    for idx in layer1_idx:
        layers_1 += [net.features[idx]]
    for idx in layer2_idx:
        layers_2 += [net.features[idx]]
    for idx in layer3_idx:
        layers_3 += [net.features[idx]]
    for idx in layer4_idx:
        layers_4 += [net.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0, layer1, layer2, layer3, layer4


def get_resnet18_layers(pretrained=True):
    from torchvision.models import resnet
    net = resnet.resnet18(pretrained=pretrained)
    layer0 = nn.Sequential(
        net.conv1, net.bn1, net.relu,
        net.maxpool
    )
    layer1 = net.layer1
    layer2 = net.layer2
    layer3 = net.layer3
    layer4 = net.layer4
    return layer0, layer1, layer2, layer3, layer4


def get_resnet34_layers(pretrained=True):
    from torchvision.models import resnet
    net = resnet.resnet18(pretrained=pretrained)
    layer0 = nn.Sequential(
        net.conv1, net.bn1, net.relu,
        net.maxpool
    )
    layer1 = net.layer1
    layer2 = net.layer2
    layer3 = net.layer3
    layer4 = net.layer4
    return layer0, layer1, layer2, layer3, layer4


def get_resnet50_layers(pretrained=True):
    from torchvision.models import resnet
    net = resnet.resnet50(pretrained=pretrained)
    layer0 = nn.Sequential(
        net.conv1, net.bn1, net.relu,
        net.maxpool
    )
    layer1 = net.layer1
    layer2 = net.layer2
    layer3 = net.layer3
    layer4 = net.layer4
    return layer0, layer1, layer2, layer3, layer4
