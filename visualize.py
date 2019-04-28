import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

from train_resnet import get_cifar10_data
from pixelcnnpp.density import density_generator

def do_samples(args):

    model = ResNet18().cuda()
    model.load_state_dict(torch.load(args.resnet_fname))
    model.eval()

    plt.figure()
    _, _, test_loader = get_cifar10_data(bs=args.bs, as_loader=True)
    real_images, real_labels = next(iter(test_loader))

    perturbed = torch.load('./data/adversarial/untargeted_step10_eps{}.pt'.format(args.eps))
    perturbed_images = perturbed['adversarials']

    for i in range(n_samples):

        real_image = real_images[i]
        real_label = real_labels[i]
        perturbed_image = perturbed_images[i]

        with torch.no_grad():
            print(model(perturbed_images[i:i+1]))
            perturbed_label = torch.max(model(perturbed_images[i:i+1]))[1]

def do_histogram(args):

    adversarials = torch.load(
        './data/adversarial/untargeted_step10_eps{}.pt'.format(args.eps)
    )['adversarials']
    adversarials = torch.from_numpy(adversarials)

    _, _, cifar10_test = get_cifar10_data(bs=None, as_loader=False)
    imgs = []
    for img, _ in cifar10_test:
        imgs.append(img)
    cifar10_test = torch.stack(imgs)

    density = density_generator()

    adversarial_densities = []
    original_densities = []

    for i in range(adversarials.shape[0]):
        x = adversarials[i:i+1].cuda()
        x = x * 2. - 1.
        with torch.no_grad():
            adversarial_densities.append(density(x).item())
        if i > 20: break

    for i in range(adversarials.shape[0]):
        x = cifar10_test[i:i+1].cuda()
        x = x * 2. - 1.
        with torch.no_grad():
            original_densities.append(density(x).item())
        if i > 20: break

    print(adversarial_densities)
    print(original_densities)

    print()

    print(sum(adversarial_densities) / len(adversarial_densities))
    print(sum(original_densities) / len(original_densities))

    pass

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--to-show', type=str, required=True)
    parser.add_argument('--resnet-fname', type=str, required=False,
        default='./logdirs/cifar10-40k_best.pt')
    parser.add_argument('--n-samples', type=int, required=False, default=3)
    parser.add_argument('--eps', type=float, required=False, default=0.0001)
    args = parser.parse_args()

    if args.to_show == 'samples':
        do_samples(args)
    elif args.to_show == 'histogram':
        do_histogram(args)
    else:
        raise RuntimeError("Invalid argument for 'to_show'")

if __name__ == '__main__':
    main()

