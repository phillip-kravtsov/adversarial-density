import time
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
from .utils import *
from .model import *
from PIL import Image



class DensityAcquisitionEngine():
    def __init__(self, model):
        self.model = model
    def get_nll(self, ims):
        ls = self.model(ims)
        return discretized_mix_logistic_loss(ims, ls, per_sample=True)
        
def get_model(load_path, device, args=None):
    if args is None:
        model = PixelCNN(nr_resnet=5, nr_filters=160, 
                    input_channels=3, nr_logistic_mix=10)
        print('model parameters loaded')
    else:
        model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
                    input_channels=3, nr_logistic_mix=args.nr_logistic_mix)
    model = model.to(device)
    model.eval()
    load_part_of_model(model, load_path, device)
    return model

def density_generator():
    load_path = 'pixelcnnpp/params/pcnn_lr.0.00040_nr-resnet5_nr-filters160_889.pth' 
    device = torch.device('cuda:0')
    model = get_model(load_path, device)
    dae = DensityAcquisitionEngine(model)
    def get_nlls(ims):
        return dae.get_nll(ims)
    return get_nlls


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data I/O
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-r', '--load_params', type=str, default=None,
                        help='Restore training from previous model checkpoint?')
    # model
    parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                        help='Number of residual blocks per stage of the model')
    parser.add_argument('-n', '--nr_filters', type=int, default=160,
                        help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                        help='Number of logistic components in the mixture. Higher = more flexible model')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size during training per GPU')
    args = parser.parse_args()
    rescaling     = lambda x : (x - .5) * 2.
    rescaling_inv = lambda x : .5 * x  + .5
    kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}
    ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
        download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    device = torch.device('cuda:0')
    model = get_model(args.load_params, device)
    dae = DensityAcquisitionEngine(model)
    with torch.no_grad():
        for im, target in train_loader:
            im = im.to(device)
            target = target.to(device)
            nlls = dae.get_nll(im)
            print(sum(nlls)/len(nlls))
            break
        #x = torch.zeros([args.batch_size, 3, 32, 32])
        #print("Simgatic tests")
        #for i, sigma in enumerate(np.logspace(-4, 0, args.batch_size, base=10)):
        #    x[i] = torch.randn([1, 3, 32, 32]) * sigma
        #x = torch.clamp(x, -1, 1)
        #x = x.to(device)
        #nll = dae.get_nll(x)
        #print("Sigma: {} NLL: {}".format(sigma, nll))
