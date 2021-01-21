import pickle
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.manifold import TSNE

def save_dict(dict,path):
    with open(path,'wb') as file:
        pickle.dump(dict,file)
    print('>>>>>>>>>>>>>>>>>>dict saved')

def load_dict(path):
    with open(path,'rb') as file:
        dict=pickle.load(file)
        print('>>>>>>>>>>>>>>>>>dict loaded')
        return dict
def save_pic_array(inputs,path,w,h):
    temp = []
    for i in range(w):
        row = torch.cat([inputs[j + i * h, :, :, :] for j in range(h)], dim=2)
        print(row.size())
        temp.append(row)
    temp = torch.cat(temp, dim=1)
    temp = temp.numpy().transpose(1, 2, 0)
    plt.imsave(path,temp)

def load_state_dict(path):

    state_dict = torch.load(path)
    new_state_dcit = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]
        else:
            name = k
        new_state_dcit[name] = v
    return new_state_dcit


def PCA_reduction(feature,dimension):
    pca = PCA(dimension)
    pca.fit(feature)
    print(pca.explained_variance_ratio_)
    print('PCA sum:', np.sum(pca.explained_variance_ratio_))
    # plt.stem(pca.explained_variance_ratio_)
    # plt.show()
    feature_new = pca.transform(feature)
    print('PCA shape:',feature_new.shape)
    return feature_new

def tSNE_show(inputs,label):
    np.random.seed(1)
    tsne=TSNE(n_components=2)
    tsne.fit(inputs)
    feature_new=tsne.embedding_
    print('tSNE shape:',feature_new.shape)
    plt.scatter(x=feature_new[:,0],y=feature_new[:,1],c=label,marker='.',linewidths=0,cmap='jet')
    plt.colorbar()
    plt.show()

def get_cosine_similarity(gate):
    eps = 1e-10
    inner_product = gate.mm(gate.t())
    f_norm = torch.norm(gate, p='fro', dim=1, keepdim=True)
    outter_product = f_norm.mm(f_norm.t())
    cosine = inner_product / (outter_product + eps)
    return cosine
