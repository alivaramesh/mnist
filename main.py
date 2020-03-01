# Code base forked from: https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import sys, os, time, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image

def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')

    parser.add_argument('--test-batch-size', type=int, default=1000, 
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--lr', type=float, default=1.0,
                        help='learning rate (default: 1.0)')

    parser.add_argument('--gamma', type=float, default=0.7,
                        help='Learning rate step gamma (default: 0.7)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')

    parser.add_argument('--log-interval', type=int, default=100,
                        help='how many batches to wait before logging training status')

    parser.add_argument('--drop_out',type=str,default= 'none', 
                        help='Drop out configuretion in form of a string specifing\
                         the layer numebr and dopout probability.\
                         Example: 6,.5,7,.25 means dropout at layers 6 and 7 with probablity .5 and .25')

    parser.add_argument('--erase_w', type=float, default = 0.,
                        help='Probablity of erasing a randomly chosen rectangular region by setting its value to 1')

    parser.add_argument('--erase_b', type=float, default = 0.,
                        help='Probablity of erasing a randomly chosen rectangular region by setting its value to 0')

    parser.add_argument('--erase_L', action='store_true',
                        help='increase the range of random regions to be erased for augmentation')
                        
    parser.add_argument('--erase_S', action='store_true',
                        help='decrease the range of random regions to be erased for augmentation')

    parser.add_argument('--rotation', type=float, default = 0.,
                        help='Range of degrees to select from for rotation augmentation. The range will be (-rotation,+rotattion)')

    parser.add_argument('--retrain', action='store_true',
                        help='Ignore existing trainings with the same config')
    
    parser.add_argument('--no-norm', action='store_true',
                        help='do not Normalize to range [0-1] and do not apply mean/var normalization')

    parser.add_argument('--batch-norm', action='store_true',
                help='Apply batch normalization')

    parser.add_argument('--ADAM', action='store_true',
                        help='Use adam optimizer with PyTorch default settings')

    parser.add_argument('--test', action='store_true',
                        help='to run test on the test variants by loading the model path provided at "model_path"')

    parser.add_argument('--model-path',type=str,default= None, 
                        help='path to an existing model')
        

    args = parser.parse_args()
    args.train = not args.test 
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.norm = not args.no_norm

    if args.train:
    
        run_label = 'run_BS_{}_EPS_{}_LR_{}_GAMA_{}'.format(args.batch_size,args.epochs,args.lr,args.gamma)
        if args.batch_norm:
            run_label += '_BN'
        if args.erase_b > 0:
            run_label += '_erase_b_{}'.format(args.erase_b)
        if args.erase_w > 0:
            run_label += '_erase_w_{}'.format(args.erase_w)

        if (args.erase_b > 0 or args.erase_w > 0):
            if args.erase_L:
                run_label += '_ERSL'
            if args.erase_S:
                run_label += '_ERSS'
            
        if args.drop_out != 'none':
            run_label += '_DRPOUT_{}'.format(args.drop_out)
        if args.ADAM:
            run_label += '_ADAM'
        if args.no_norm:
            run_label += '_NONORM'
        if args.rotation != 0:
            run_label += '_ROT_{}'.format(args.rotation)
            
        existing = os.listdir('exps')
        existing = [d for d in existing if d.startswith('run')]
        existing = {'_'.join(d.split('_')[:-1]) for d in existing}
        if not args.retrain and run_label in existing:
            print('Model "{}" already exists'.format(run_label))
            sys.exit(0)

        run_label += '_{}'.format(int(time.time()))
        
        args.save_dir = os.path.join('exps',run_label)
    else:
        run_label = '{}_test_{}'.format(args.model_path,int(time.time()))
        if args.no_norm:
            run_label += '_no_norm'
        args.save_dir = run_label
    
    if args.train and args.drop_out != 'none':
        tmp = args.drop_out.split(',')
        args.drop_out = { int(tmp[2*i]):float(tmp[2*i+1]) for i in range(len(tmp)//2)}
    else:
        args.drop_out={}

    os.mkdir(args.save_dir)
    command='Command: ' + ' '.join(sys.argv)+'\n'
    print(command)
    print("save_dir:",args.save_dir)
    with open(os.path.join(args.save_dir,'log'),'w') as of:
        of.write(command)
    return args

class NumPyDataset(torch.utils.data.Dataset):
    def __init__(self, npypath,base_trans,norm):
        data = np.load(npypath, allow_pickle=True).item()
        self.data, self.target = data['x'], data['y']
        print("NumPyDataset loaded; x.shape: {} y.shape: {}".format(self.data.shape,self.target.shape))
        self.norm = norm
        self.transforms = transforms.Compose(base_trans)
        
    def __getitem__(self, index):
        x = self.data[index]
        if self.norm:
            x = (x*255/(x.max()+1e-8) ).astype(np.uint8)
            x = Image.fromarray(x.reshape(28,28), 'L')
        else:
            x = Image.fromarray(x.reshape(28,28), 'F')
        x = self.transforms(x)
        y = self.target[index]

        return x, y

    def __len__(self):
        return len(self.target)

class Net(nn.Module):
    def __init__(self,drop_out={},bn=False):
        super(Net, self).__init__()

        layer_dims = [(28*28,500),(500, 500),(500, 500),
                      (500, 200),(200, 200),
                      (200,100),
                      (100,50),
                      (50,10)]
        layers = []
        for l in range(7):
            layers.extend([nn.Linear(*layer_dims[l],bias= l<=4 or not bn),nn.ReLU()])
            if bn and l>4:
                layers.append(nn.BatchNorm1d(num_features=layer_dims[l][1]))
            if l+1 in drop_out:
                layers.append(nn.Dropout2d(drop_out[l+1]))
        layers.append(nn.Linear(*layer_dims[-1]))
        self.net = torch.nn.Sequential(*layers)


    def forward(self, x):

        output = self.net(x.reshape((-1,28*28)))
        output = F.log_softmax(output, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss=0
    for batch_idx, (data, target) in enumerate(train_loader):
        # if batch_idx > 10:
        #     break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    train_loss /= (batch_idx)

    print('Train Average loss: {:.4f}'.format(train_loss))
    return train_loss

def test(args, model, device, test_loader, variant):
    model.eval()
    test_loss = 0
    correct = 0
    pred = []
    true=[]
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            _pred = output.argmax(dim=1, keepdim=True)
            pred.extend(_pred.cpu().numpy())
            true.extend(target.cpu().numpy())
            correct += _pred.eq(target.view_as(_pred)).sum().item()

    test_loss /= len(test_loader)
    acc= 100. * correct / len(test_loader.dataset)
    print('Test on {} variant: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(variant,
        test_loss, correct, len(test_loader.dataset),acc))
    pred =  np.array(pred)
    true =  np.array(true)
    pred = np.concatenate([true.reshape((-1,1)),pred],1)
    return acc, test_loss, pred

def main():
    
    args = get_args()

    device = torch.device("cuda" if args.use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.use_cuda else {}
    base_trans = [transforms.ToTensor()]
    base_trans.append(transforms.Normalize((0.1307,), (0.3081,)))
    
    if args.train:
        train_trans = [t for t in base_trans]
        erase_trans = []
        if args.erase_L:
            erase_trans.append(transforms.RandomErasing(p=args.erase_b, scale=(0.02, 0.6), ratio=(0.3, 4.3), value=0))
            erase_trans.append(transforms.RandomErasing(p=args.erase_w, scale=(0.02, 0.6), ratio=(0.3, 4.3), value=1))
        elif args.erase_S:
            erase_trans.append(transforms.RandomErasing(p=args.erase_b, scale=(0.02, 0.2), ratio=(0.3, 2.3), value=0))
            erase_trans.append(transforms.RandomErasing(p=args.erase_w, scale=(0.02, 0.2), ratio=(0.3, 2.3), value=1))
        else:
            erase_trans.append(transforms.RandomErasing(p=args.erase_b, scale=(0.02, 0.4), ratio=(0.3, 3.3), value=0))
            erase_trans.append(transforms.RandomErasing(p=args.erase_w, scale=(0.02, 0.4), ratio=(0.3, 3.3), value=1))
        erase_trans = transforms.RandomChoice(erase_trans)
        train_trans.insert(1,erase_trans)

        train_trans.insert(0,transforms.RandomAffine(args.rotation))

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                        transform=transforms.Compose(train_trans)),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('data', train=False, transform=transforms.Compose(base_trans)),
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)

    test_variants = ["clean", "t1", "t2", "t3", "t4"]
    test_loaders = {t:torch.utils.data.DataLoader(
        NumPyDataset('data/test_sets/{}.npy'.format(t),base_trans,args.norm), 
        batch_size=args.test_batch_size, shuffle=True, **kwargs) for t in test_variants}
    
    model = Net(drop_out=args.drop_out,bn=args.batch_norm).to(device)
    epoch_logs = {}
    if args.train:
        optimfn = optim.Adam if args.ADAM else optim.Adadelta
        optimizer = optimfn(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            tr_loss = train(args, model, device, train_loader, optimizer, epoch)
            epoch_logs.update({epoch: {'train':(tr_loss,) }})
            # test 
            # test(args, model, device, test_loader, 'orig_test')
            for t in test_loaders:
                acc, ts_loss, preds, = test(args, model, device, test_loaders[t],t)  
                epoch_logs[epoch].update({t:(ts_loss,acc)})
            scheduler.step()
        
        torch.save(model.state_dict(), os.path.join(args.save_dir,"mnist.pth"))

        plot_res(args,test_variants,epoch_logs) 
    else:
        model.load_state_dict(torch.load(args.model_path))
        # test 
        for t in test_loaders:
            acc, ts_loss, preds, = test(args, model, device, test_loaders[t],t)  
            epoch_logs.setdefault('final',{}).update({t:(ts_loss,acc)})
            np.save(os.path.join(args.save_dir,'preds_{}.npy'.format(t)),preds)
    log(args,test_variants,epoch_logs)    

def log(args,test_variants,epoch_logs):
    '''
        Save the logs
    '''
    with open(os.path.join(args.save_dir,'log'),'a') as of:
        for epoch in sorted(epoch_logs.keys()):
            tmp = '{}'.format(epoch)
            if args.train:
                tmp += ',{},'.format(epoch,epoch_logs[epoch]['train'][0])
            tmp += ','+','.join( map(str, [ epoch_logs[epoch][t][0] for t in test_variants] ))
            tmp += ','+','.join( map(str, [ epoch_logs[epoch][t][1] for t in test_variants] ))
            of.write(tmp+'\n')

def plot_res(args,test_variants,epoch_logs) :
    '''
        Plot the resulst 
    '''
    losses = {'train':[epoch_logs[e]['train'][0] for e in range(1, args.epochs + 1)]}
    losses.update({t : [epoch_logs[e][t][0] for e in range(1, args.epochs + 1)] for t in test_variants})
    test_acc = {t : [epoch_logs[e][t][1] for e in range(1, args.epochs + 1)] for t in test_variants}

    fig,[ax1,ax2] = plt.subplots(1,2)
    
    cmap = plt.cm.get_cmap('jet', len(losses))
    colors = {t:cmap(i)  for i,t in enumerate(losses.keys())}
    for i,t in enumerate(losses.keys()):
        linestyle = 'solid' if t == 'train' else 'dashed'
        ax1.plot(range(1, args.epochs + 1), losses[t], label = t, color=colors[t],linestyle = linestyle)
    for i,t in enumerate(test_acc.keys()):
        ax2.plot(range(1, args.epochs + 1), test_acc[t], color=colors[t])
    ax1.set_xlabel("Training epochs")
    ax2.set_xlabel("Training epochs")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    ax1.set_title("Train sn test Loss")
    ax2.set_title("Accuracy on test variants")
    ax1.grid()
    ax2.grid()
    fig.legend()
    fig.set_size_inches(18,9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.save_dir, 'results.jpg'))

if __name__ == '__main__':
    main()