
import sys, os

import numpy as np


import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}
matplotlib.rc('font', **font)

import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
from torchvision import datasets, transforms

from collections import defaultdict

from PIL import Image

class NumPyDataset(torch.utils.data.Dataset):
    def __init__(self, npypath):
        data = np.load(npypath, allow_pickle=True).item()
        self.data, self.target = data['x'], data['y']
        print("NumPyDataset loaded; x.shape: {} y.shape: {}".format(self.data.shape,self.target.shape))

        self.transforms = transforms.Compose([
                           transforms.ToTensor(),])
                           transforms.Normalize((0.1307,), (0.3081,))])
    def __getitem__(self, index):
        x = self.data[index]
        x = (x*255/x.max()).astype(np.uint8)
        x = Image.fromarray(x.reshape(28,28), 'F')
        x = self.transforms(x)
        y = self.target[index]

        return x, y

    def __len__(self):
        return len(self.target)

if __name__ == '__main__':

    use_cuda = 0

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    base_trans = [transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]
    train_trans = [t for t in base_trans]
    erase_trans = []
    erase_trans.append(transforms.RandomErasing(p=0.8, scale=(0.02, 0.4), ratio=(0.3, 3.3), value=0))
    erase_trans.append(transforms.RandomErasing(p=0.8, scale=(0.02, 0.4), ratio=(0.3, 3.3), value=1))
    erase_trans = transforms.RandomChoice(erase_trans)
    train_trans.insert(1,erase_trans)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose(train_trans)),
        batch_size=1, shuffle=False, **kwargs)

    # nvis = 20
    labels  = defaultdict(list)
    # for i,batch in enumerate(train_loader):
    #     if i < 40:
    #         _im = plt.imshow(batch[0][0][0].numpy())
    #         divider = make_axes_locatable(plt.gca())
    #         cax = divider.append_axes("right", size="5%", pad=0.05)
    #         plt.gcf().colorbar(_im, cax=cax)
    #         plt.savefig('vis/samples/samples_train_{}_{}.jpg'.format(i,batch[1][0].numpy()), bbox_inches = 'tight',pad_inches = 0)
    #         plt.close()

    #     labels['train'].append(batch[1][0].numpy())

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ])),batch_size=1, shuffle=False, **kwargs)

    npytest_loader = torch.utils.data.DataLoader(
        NumPyDataset('data/test_sets/clean.npy'), 
        batch_size=1, shuffle=True, **kwargs)
    print('len(npytest_loader):',len(npytest_loader))

    # for i,batch in enumerate(test_loader):
    #     if i < nvis:
    #         # print("batch[0].shape",batch[0].shape)
    #         plt.imshow(batch[0][0][0].numpy())
    #         plt.savefig('vis/samples_test_{}_{}.jpg'.format(i,batch[1][0].numpy()))
    #     labels['test'].append(batch[1][0].numpy())

    # for i,batch in enumerate(npytest_loader):
    #     if i < nvis:
    #         # print("batch[0].shape",batch[0].shape)
    #         plt.imshow(batch[0][0][0].numpy())
    #         plt.savefig('vis/samples_npytest_{}_{}.jpg'.format(i,batch[1][0].numpy()))
    #     labels['npytest_loader'].append(batch[1][0].numpy())

    intenses = defaultdict(list)
    for t in ["clean", "t1", "t2", "t3", "t4"]:
        data = np.load("data/test_sets/" + t + ".npy", allow_pickle=True).item()
        x, y = data['x'], data['y']
        tmp = x.reshape((-1,))
        # tmp = (tmp-tmp.min()) / (tmp.max() - tmp.min())
        intenses[t].extend(tmp.tolist())
        labels[t].extend(y)
    #     for i in range(20):
    #         _im = plt.imshow(x[i].squeeze())
    #         divider = make_axes_locatable(plt.gca())
    #         cax = divider.append_axes("right", size="5%", pad=0.05)
    #         plt.gcf().colorbar(_im, cax=cax)
    #         plt.savefig('vis/samples/samples_{}_{}_{}.jpg'.format(t,i,y[i]), bbox_inches = 'tight',pad_inches = 0)
    #         plt.close()
    #eq=np.array(sorted(intenses['clean'])) == np.array(sorted(intenses['t1'])) 
    eq = np.round(intenses['clean'],7) == np.round(intenses['t1'],7) 
    print( np.all(eq), np.sum(eq))
    # for i,b in enumerate(eq):
    #     if not b:
    #         print(intenses['clean'][i], intenses['t1'][i])
    eq = np.array(intenses['clean']) == np.array(intenses['t2']) 
    print( np.all(eq), np.sum(eq))

    for s in labels:
        fig,axs = plt.subplots(1,1)
        print(s, len(labels[s]))
        axs.hist(labels[s],bins=100, density=True)
        axs.set_title(s)
        axs.set_xticks(range(1,11,1))
        axs.set_xticklabels(map(str,range(1,11,1)))
        axs.set_xlabel("Categories")
        axs.set_ylabel("Density")
        axs.grid()
        fig.tight_layout()
        fig.set_size_inches(6,4)
        fig.savefig('vis/hist_label_{}.jpg'.format(s))

    hists = {}
    for s in labels:
        fig,axs = plt.subplots(1,1)
        print(s, len(labels[s]))
        h,_,_=axs.hist(intenses[s],bins=100, density=True)
        hists.update({s:h})
        axs.set_title(s)
        axs.set_xlabel("Intensity")
        axs.set_ylabel("Density")
        axs.grid()
        fig.tight_layout()
        fig.set_size_inches(6,4)
        fig.savefig('vis/hist_intens_nonorm_{}.jpg'.format(s))

    print("Manahatan distances")
    for s in ["t1", "t2", "t3", "t4"]:
        mandist = np.sum(hists['clean'] - hists[s])
        print(s,mandist)

