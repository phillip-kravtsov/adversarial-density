import argparse
import os
import numpy as np

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

from helpers.loaders import getwmloader, _getdatatransformsdb
from helpers.utils import progress_bar
import torchvision.transforms as transforms
import scipy
import scipy.misc
import cv2
import matplotlib.pyplot as plt

def get_prediction(inp, model):
    return model(inp.squeeze().unsqueeze(0)).squeeze().argmax(dim=0).item()

def get_error(v, X, model, device, bs=64):

    loader = torch.utils.data.DataLoader(
        X, batch_size=bs, shuffle=False, num_workers=4)

    n_correct, n_total = 0, 0

    for inp, _ in loader:
        inp = inp.to(device)
        n_correct += \
            (torch.argmax(model(inp), dim=1) == torch.argmax(model(inp + v), dim=1)).sum().item()
        n_total += inp.shape[0]

    return 1. - n_correct / float(n_total)

def get_l2_update(inp, v, model, device, stepsize):

    orig_pred = get_prediction(inp, model)
    r = torch.zeros(3, 32, 32).to(device)
    r.requires_grad = True

    while get_prediction(inp + v + r, model) == orig_pred:

        #print(model((inp + v + r).unsqueeze(0)).squeeze()[orig_pred].item())
        
        grad = torch.autograd.grad(
            model((inp + v + r).unsqueeze(0)).squeeze()[orig_pred],
            [r],
        )[0]
        r = r.detach() - stepsize * grad
        r = r.detach()
        r.requires_grad = True

    return r

def project(sample, p, xi, device):

    assert sample.shape == (3, 32, 32)

    if p == 2.:
        sample = sample * min(1., xi / torch.norm(sample.reshape(-1)))
    elif p == float('inf'):
        sample = sample.sign() * torch.min(
            torch.abs(sample), xi * torch.ones(*sample.shape).to(device))
    else:
        raise RuntimeError("Unsupported p")

    return sample

def get_universal_perturbation(X, model, p, xi, delta, l2_update_stepsize, device):

    v = torch.zeros(3, 32, 32).to(device)

    while get_error(v, X, model, device) <= 1. - delta:

        for inp, out in X:

            inp = inp.to(device)
            if get_prediction(inp, model) != get_prediction(inp + v, model):
                continue

            l2_update = get_l2_update(inp, v, model, device, l2_update_stepsize)
            v = project(v + l2_update, p, xi, device)

    return v

def do_adversarial(args):

    batch_size = 1
    wmloader, wmdataset = getwmloader(args.wm_path, batch_size, args.wm_lbl)

    transform_train, transform_test = _getdatatransformsdb(datatype='cifar10')
    trainset = torchvision.datasets.CIFAR10(
        root=args.db_path, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=args.db_path, train=False, download=True, transform=transform_test)

    wmset = [tup for tup in wmdataset]
    del transform_train, transform_test, wmloader, wmdataset
    
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.exists(args.model_path), 'Error: no checkpoint found!'
    checkpoint = torch.load(args.model_path)
    net = checkpoint['net']
    acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    net.eval()

    train_perturbation = get_universal_perturbation(
        [trainset[i] for i in range(args.len_train_x)],
        net, args.p, args.xi, args.delta, args.l2_update_stepsize, device
    )

    import matplotlib.pyplot as plt
    tp = train_perturbation.detach().cpu().numpy()
    tp = (tp - np.min(tp)) / (np.max(tp) - np.min(tp))
    tp = np.moveaxis(tp, 0, -1)
    plt.imshow(tp)
    plt.show()

    wm_perturbation = get_universal_perturbation(
        [wmset[i] for i in range(args.len_wm_x)],
        net, args.p, args.xi, args.delta, args.l2_update_stepsize, device
    )

    tp = wm_perturbation.detach().cpu().numpy()
    tp = (tp - np.min(tp)) / (np.max(tp) - np.min(tp))
    tp = np.moveaxis(tp, 0, -1)
    plt.imshow(tp)
    plt.show()
    


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='checkpoint/teacher-cifar100-2.t7', help='the model path')
    parser.add_argument('--wm_path', default='./data/trigger_set/', help='the path the wm set')
    parser.add_argument('--wm_lbl', default='labels-cifar.txt', help='the path the wm random labels')
    parser.add_argument('--db_path', default='./data', help='the path to the root folder of the test data')
    parser.add_argument('--p', type=float, required=True)
    parser.add_argument('--xi', type=float, required=True)
    parser.add_argument('--delta', type=float, required=True)
    parser.add_argument('--len-train-x', type=int, required=True)
    parser.add_argument('--len-wm-x', type=int, required=True)
    parser.add_argument('--l2-update-stepsize', type=float, required=False, default=0.01)

    args = parser.parse_args()
    print(args)

    assert args.p >= 1.
    do_adversarial(args)

if __name__ == '__main__':
    main()

