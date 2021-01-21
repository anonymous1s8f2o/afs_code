import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.autograd as ta
from collections import OrderedDict
from tqdm import tqdm
import torch

import argparse

parser = argparse.ArgumentParser(description='adversarial feature stacking')
parser.add_argument('--pert', type=int)
args = parser.parse_args()

PERT_SIZE=args.pert
lr=0.1
batch_size = 128
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
root_path='exp/brothers_preact_resnet18/brothers_{}'.format(PERT_SIZE)


decay=[50,60,150,200,10000]

if(os.path.exists(root_path)==False):
    os.makedirs(root_path)

print(root_path)
transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

# import models.wideresnet as model
import models.preact_resnet as model

net=model.PreActResNet18().cuda()

criterion_cross_entropy = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

def adjust_lr(epoch,lr):
    if((1+epoch)%decay[0]==0):
        lr/=10
        del decay[0]
    return lr

def get_adv_inputs_pert(net, image, label, step_size,pert, iteration):
    delta=(torch.rand_like(image)-torch.rand_like(image)).cuda()*pert
    image = image.cuda()
    label = label.cuda()
    for i in range(iteration):
        temp = image + delta
        temp = temp.requires_grad_(True)
        outputs= net(temp)
        loss = criterion_cross_entropy(outputs, label)
        grad = ta.grad(loss, temp, torch.ones_like(loss).cuda())[0]
        delta=delta+torch.sign(grad) * step_size
        delta[delta>pert]=pert
        delta[delta<-pert]=-pert
    image = image + delta
    image[image > 1] = 1
    image[image<0]=0
    return image

def train(epoch):
    net.train()
    global lr
    lr = adjust_lr(epoch,lr)
    for p in optimizer.param_groups:
        p['lr'] = lr
    print('\nEpoch: %d,lr: %.5f  ' % (epoch, lr),root_path)
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs=get_adv_inputs_pert(net,inputs,targets,2/255,PERT_SIZE/255,10)

        optimizer.zero_grad()
        pred = net(inputs)
        loss = criterion_cross_entropy(pred, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        indicator=int(len(trainloader)/2)
        if ((batch_idx+1)%indicator == 0):
            print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss / (batch_idx + 1),100. * correct / total, correct,total))
def main():
    for epoch in range(65):
        train(epoch)
        torch.save(net.state_dict(), root_path + '/parameter_opt_{}.pkl'.format(epoch))
main()
