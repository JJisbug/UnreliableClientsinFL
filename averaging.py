#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
from torch import nn


def average_weights(w, server_acc):
    w_avg = copy.deepcopy(w[0])# first user's w_t
    for k in w_avg.keys():
        for i in range(0, len(w)):
            if i == 0:
                w_avg[k] *= server_acc[i]
            else:
                w_avg[k] += w[i][k]*server_acc[i]
        #w_avg[k] = torch.div(w_avg[k], len(w))
    # for this_key in w_avg.keys():
    #     dev = np.random.normal(0, 0.1, w_avg[this_key].size())
    #     dev = torch.from_numpy(dev).float().cuda()
    #     w_avg[this_key] = w_avg[this_key] + dev
    return w_avg
