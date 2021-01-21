'''
Modified from https://github.com/pytorch/vision.git
'''

import torch
import torch.nn as nn
class net(nn.Module):
    '''
    VGG model
    '''
    def __init__(self,channel,numclass):
        super(net, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(channel, numclass),
        )
    def forward(self, x):
        x = self.classifier(x)
        return x
