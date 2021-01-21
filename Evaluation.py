import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.autograd as ta
from tqdm import tqdm
import torchvision
from autoattack import AutoAttack
torch.backends.cudnn.benchmark = True
class Evaluation:
    def __init__(self,dataset,net,batch_size):
        print('evaluate: batch_size:{}'.format(batch_size),dataset)
        self.dataset=dataset
        self.net=net
        self.batch_size=batch_size
        self.testloader=None
        self.criterion_cross_entropy = nn.CrossEntropyLoss().cuda()

    def load_data_loader(self):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        if(self.dataset=='cifar10'):
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        elif(self.dataset=='cifar100'):
            testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        print('dataloader loaded :',self.dataset)

    def PGD_attack(self,image, label, step_size, pert, iteration):
        delta = (torch.rand_like(image) - torch.rand_like(image)).cuda() * pert
        image = image.cuda()
        label = label.cuda()
        for i in range(iteration):
            temp = image + delta
            temp = temp.requires_grad_(True)
            outputs = self.net(temp)
            loss = self.criterion_cross_entropy(outputs, label)
            grad = ta.grad(loss, temp, torch.ones_like(loss).cuda())[0]
            delta = delta + torch.sign(grad) * step_size
            delta[delta > pert] = pert
            delta[delta < -pert] = -pert
        image = image + delta
        image[image > 1] = 1
        image[image < 0] = 0
        return image

    def FGSM_attack(self,image, label, pert):
        image = image.cuda()
        label = label.cuda()
        temp = image
        temp = temp.requires_grad_(True)
        outputs = self.net(temp)
        loss = self.criterion_cross_entropy(outputs, label)
        grad = ta.grad(loss, temp, torch.ones_like(loss).cuda())[0]
        delta = torch.sign(grad) * pert
        delta[delta > pert] = pert
        delta[delta < -pert] = -pert
        image = image + delta
        image[image > 1] = 1
        image[image < 0] = 0
        return image

    def evaluate_accuracy(self):
        print('evaluate_accuracy------------------------')
        total, correct,test_loss = 0, 0,0
        for batch_idx, (inputs, targets) in tqdm(enumerate(self.testloader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            pred = self.net(inputs)
            _, predicted = pred.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            loss = self.criterion_cross_entropy(pred, targets)
            test_loss += loss.item()
        acc = correct / total
        print('benign accuracy:{}, loss:{}'.format(acc,test_loss/(batch_idx+1)))
        return acc

    def evaluate_PGD(self,pert,iter):
        print('evaluate_PGD------------------------')
        total, correct,test_loss = 0, 0,0
        for batch_idx, (inputs, targets) in tqdm(enumerate(self.testloader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs=self.PGD_attack(inputs,targets,2/255,pert,iter)
            pred = self.net(inputs)
            _, predicted = pred.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            loss = self.criterion_cross_entropy(pred, targets)
            test_loss += loss.item()
        acc = correct / total
        print('PGD accuracy:{},loss:{}, iters:{},pert:{}'.format(acc,test_loss/(batch_idx+1),iter,pert))
        return acc

    def evaluate_FGSM(self,pert):
        print('evaluate_FGSM------------------------')
        total, correct = 0, 0
        for batch_idx, (inputs, targets) in tqdm(enumerate(self.testloader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs=self.FGSM_attack(inputs,targets,pert)
            pred = self.net(inputs)
            _, predicted = pred.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        acc = correct / total
        print('FGSM accuracy:{},pert:{}'.format(acc,pert))
        return acc

    def evaluate_autoattack(self,pert,batch_size_aa):
        print('evaluate_autoattack------------------------')
        print('pert:{}'.format(pert))
        adversary = AutoAttack(self.net, norm='Linf', eps=pert, version='standard')
        # adversary.attacks_to_run = ['apgd-t']
        for batch_idx, (inputs, targets) in tqdm(enumerate(self.testloader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            adversary.run_standard_evaluation(inputs, targets, batch_size_aa)
            print('autoattack_done')
            break

    def test_robust_score_PGD(self,start,stop,num,iter):
        print('testing: start:{},stop:{},num:{}'.format(start,stop,num))
        perburbation = np.linspace(start, stop, num)
        score = []
        for pert in perburbation:
            acc = self.evaluate_PGD(pert,iter)
            score.append(acc)
            print('pert={},acc={}'.format(pert,acc))
        print('score:', np.sum(score))
        return np.sum(score)



