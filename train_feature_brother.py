import os
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.autograd as ta
from tqdm import tqdm
import torch
import Brothers
import models.linear_classifier as model_l
import argparse

parser = argparse.ArgumentParser(description='adversarial feature stacking')
parser.add_argument('--root_path',type=str)
parser.add_argument('--weights', nargs='+', type=int)
parser.add_argument('--ratio', type=float)
args = parser.parse_args()

lr=0.1
batch_size = 100
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'


decay=[3,6,9,10000]

if(os.path.exists(args.root_path)==False):
    os.makedirs(args.root_path)
print(args.root_path,args.ratio)

transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
])
transform_test = transforms.Compose([ 
            transforms.ToTensor(),
]) 

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=True, num_workers=4)

classifier=model_l.net(channel=640*sum(args.weights)).cuda()
brother=Brothers.Brothers(dataset='cifar10',weights=[0,0,0,1,1,1,1,1,1],gpu_num=4)
brother.load_brothers()

criterion_cross_entropy = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

def adjust_lr(epoch,lr):
    if((1+epoch)%decay[0]==0):
        lr/=10
        del decay[0]
    return lr

def net_forward(inputs):
    feature = brother.brother_feature_forward(inputs)
    pred = classifier(feature)
    return pred

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
    classifier.train() 
    global lr
    lr = adjust_lr(epoch,lr)
    for p in optimizer.param_groups:
        p['lr'] = lr
    print('\nEpoch: %d,lr: %.5f  ' % (epoch, lr),args.root_path)
    train_loss = 0
    correct = 0
    total = 0
    half_batch = int(batch_size*args.ratio)
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        if(half_batch<batch_size):
            inputs_adv=get_adv_inputs_pert(net_forward,inputs[half_batch:,:],targets[half_batch:],2/255,8/255,10)
            inputs=torch.cat([inputs[:half_batch,:],inputs_adv],dim=0).detach()
        optimizer.zero_grad()
        pred = net_forward(inputs)
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


def test(adv):
    classifier.eval()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        if(adv==True):
            inputs = get_adv_inputs_pert(net_forward, inputs, targets, 2 / 255, 8/ 255, 10)
        pred=net_forward(inputs)
        _, predicted = pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss / (batch_idx + 1),100. * correct / total, correct,total))
def main():
    for epoch in range(7):
        train(epoch)
        torch.save(classifier.state_dict(), args.root_path + '/parameter_opt_{}.pkl'.format(epoch))
        test(False)
        test(True)
main()
