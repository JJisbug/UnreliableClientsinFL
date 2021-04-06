# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 12:55:08 2020

@author: MI
"""

#测试Pytorch是否下载成功
import torch
#测试cuda能否使用，能使用则返回True
print("is GPU:",torch.cuda.is_available())

#测试cuDNN是否正常，正常返回True
from torch.backends import cudnn
a = torch.tensor(1.)
print("cuDNN is OK:", cudnn.is_acceptable(a.cuda()))
