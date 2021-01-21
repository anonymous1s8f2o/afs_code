'''
Modified from https://github.com/pytorch/vision.git
'''
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
channel=1024
class net(nn.Module):
    '''
    VGG model
    '''
    def __init__(self):
        super(net, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(32*32*3,channel),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(channel, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        features = self.features(x)
        x = self.classifier(features)
        return x, features

def gradient_wrt_image(loss,image):
    grad=autograd.grad(loss,image,grad_outputs=torch.ones_like(loss).cuda(),retain_graph=True,create_graph=True)[0]
    return grad

def mask_grad_regularizer(grad,mask):
    mask[mask>0.5]=0.5
    grad=torch.abs(grad)
    result=grad*(1-mask)
    result=result.mean()
    return result
