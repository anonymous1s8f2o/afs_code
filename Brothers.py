import torch
import torch.nn as nn
from collections import OrderedDict
import models.wideresnet as model_w
import models.linear_classifier as model_l
torch.backends.cudnn.benchmark = True
class Brothers:
    def __init__(self,dataset,weights,gpu_num):
        self.NET=None
        self.classifier=None
        self.weights=weights
        self.dataset=dataset
        self.gpu_num=gpu_num
        if(self.dataset=='cifar10'):
            self.numclass=10
            self.root_path = 'exp/wideresnet2810_cifar10/'
        elif(self.dataset=='cifar100'):
            self.numclass=100
            self.root_path = 'exp/wideresnet2810_cifar100/'

    def load_state_dict(self,path):
        state_dict = torch.load(path)
        new_state_dcit = OrderedDict()
        for k, v in state_dict.items():
            if 'module' in k:
                name = k[7:]
            else:
                name = k
            new_state_dcit[name] = v
        return new_state_dcit

    def load_brothers(self):
        NET = []
        for i in range(9):
            if(self.weights[i]==1):
                net = model_w.net(num_classes=self.numclass).cuda()
                net.eval()
                path = self.root_path + 'model/brothers_{}/parameter_opt_63.pkl'.format(i)
                para = self.load_state_dict(path)
                net.load_state_dict(para)
                if(self.gpu_num>1):
                    net = nn.DataParallel(net, device_ids=range(self.gpu_num))
                NET.append(net)
        self.NET=NET
        print('net loaded:',len(NET))
        return NET

    def load_classifier(self):
        net=model_l.net(channel=640*sum(self.weights),numclass=self.numclass).cuda()
        path_name=[str(item) for item in self.weights]
        path_name=''.join(path_name)
        net.eval()
        path = self.root_path + 'classifier_ensemble/'+path_name+'/parameter_opt_4.pkl'
        para = self.load_state_dict(path)
        net.load_state_dict(para)
        if (self.gpu_num > 1):
            net = nn.DataParallel(net, device_ids=range(self.gpu_num))
        self.classifier=net
        print('classifier loaded: ',path_name)
        return net

    def load_classifier_ratio(self,file_name):
        net=model_l.net(channel=640*sum(self.weights),numclass=self.numclass).cuda()
        net.eval()
        path = self.root_path + 'classifier_ratio/'+file_name+'/parameter_opt_4.pkl'
        para = self.load_state_dict(path)
        net.load_state_dict(para)
        if (self.gpu_num > 1):
            net = nn.DataParallel(net, device_ids=range(self.gpu_num))
        self.classifier=net
        print('classifier loaded: ',file_name)
        return net

    def brother_feature_forward(self,inputs):
        PRED = []
        for net in self.NET:
            pred = net(inputs, hidden=True)
            PRED.append(pred)
        feature = torch.cat(PRED, dim=1)
        return feature

    def feature_classifier_joint_forward(self,inputs):
        PRED = []
        for net in self.NET:
            pred = net(inputs, hidden=True)
            PRED.append(pred)
        feature = torch.cat(PRED, dim=1)
        pred = self.classifier(feature)
        return pred

    def logit_ensemble_forward(self,inputs):
        PRED = []
        for net in self.NET:
            pred = net(inputs, hidden=False)
            PRED.append(pred)
        pred = torch.stack(PRED, dim=0).mean(dim=0)
        return pred
