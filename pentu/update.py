

'''
Author: Haoyu
Date: 2021-07-27 09:45:12
'''
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import  Variable


import dataset


class Memory_network(nn.Module):
    '''
        Use memory network to update support feature matrix    
    '''

    def __init__(self, feat_channel, matrix_size):
        super(Memory_network, self).__init__()
        self.feat_matrix=None
        self.matrix_size = matrix_size
        self.feat_channel = feat_channel
        self.fc_write = nn.Linear(
            in_features=feat_channel*2, out_features=matrix_size)
        self.fc_read = nn.Linear(
            in_features=feat_channel, out_features=matrix_size)

    def renew_feat_matrix(self, supp_feat, query_feat):
        '''
            @description: depend on current feat_matrix size choose update method
            @param: input supp_feat
            @return: None
        '''        
        if (self.feat_matrix is None):
            supp_feat = supp_feat.unsqueeze(1)
            # self.feat_matrix = Variable(torch.zeros_like(supp_feat), requires_grad = True)
            self.feat_matrix = supp_feat
            self.feat_matrix.requires_grad = True
            return supp_feat
        elif(self.feat_matrix.shape[1] < self.matrix_size):
            self.feat_matrix = self.write_insert(
                supp_feat, self.feat_matrix)
            return supp_feat
        else:
            self.feat_matrix = self.write_update(
                supp_feat, self.feat_matrix)
            read_out=self.read(query_feat, self.feat_matrix)
            read_out.requires_grad = True
            return read_out
    def write_insert(self, supp_feat, feat_matrix):
        if (feat_matrix.shape[1] >= self.matrix_size):
            return feat_matrix
        supp_feat = supp_feat.unsqueeze(1)
        feat_matrix = torch.cat([supp_feat, feat_matrix], 1)
        return feat_matrix

    def write_update(self, supp_feat, feat_matrix):
        '''
            description: use current support feature update feature matrix
            param  supp_feat : (n,2d,1,1)
            param  feat_matrix : (n,m,2d,1,1)
        '''
        n, m, _, __, ___ = feat_matrix.shape
        fc_write_in = supp_feat.view(n, -1)  # (n,2d)
        write_weight = self.fc_write(fc_write_in)  # (n,m)
        write_weight = F.softmax(write_weight, dim=1)
        write_weight = write_weight.reshape(n, m, 1, 1, 1)
        supp_feat = supp_feat.unsqueeze(1)  # (n,1,2d,1,1)
        supp_feat = supp_feat.expand(n, m, -1, 1, 1)  # (n,m,2d,1,1)
        feat_matrix = (1-write_weight)*feat_matrix+write_weight*supp_feat

        return feat_matrix

    def read(self, query_feat, feat_matrix):
        '''
            description: 
            param query_feat : (n,d,1,1)
            return feat_matrix: (n,m,2d,1,1)
        '''
        n = query_feat.shape[0]
        read_weight_in = query_feat.view(n, -1)
        read_weight = self.fc_read(read_weight_in)  # (n,m)
        read_weight = F.softmax(read_weight, dim=1)  # (n,m)
        read_weight = read_weight.reshape(n, self.matrix_size, 1, 1, 1)  # (n,m,1,1,1)
        read_out = (feat_matrix * read_weight).sum(1)  # (n,2d,1,1)
        return read_out


def test():
    memory = Memory_network(10, 10)

    d = 10
    query_feat = torch.rand((8, 10, 1, 1))
    sup_feat = torch.rand((8, 20, 1, 1))
    fea_matrix = torch.ones(8, 5, 20, 1, 1)
    for i in range(16):
        fea_matrix = memory.write_insert(sup_feat, fea_matrix)
    ma = memory.write_update(sup_feat, fea_matrix)

    read = memory.read(query_feat, ma)


if __name__ == '__main__':
    exit(test())
