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
channel=64
class net(nn.Module):
    '''
    VGG model
    '''
    def __init__(self):
        super(net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,channel,3,1,1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(channel, channel*2, 3, 1, 1),
            nn.BatchNorm2d(channel*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channel*2,channel*2, 3, 1, 1),
            nn.BatchNorm2d(channel*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),


            nn.Conv2d(channel*2, channel*4, 3, 1, 1),
            nn.BatchNorm2d(channel*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channel*4, channel*4, 3, 1, 1),
            nn.BatchNorm2d(channel*4),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(channel*4, 10),

        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        x= nn.AdaptiveAvgPool2d((1,1))(features)
        x =  x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, features
