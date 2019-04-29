import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

from train_resnet import get_cifar10_data
from resnet import ResNet18
from pixelcnnpp.density import density_generator

CIFAR_CLASSES = (
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
)

def get_real_adv_proj(eps):

    _, _, test = get_cifar10_data(bs=256, as_loader=True)
    images, labels = [], []
    for images_, labels_ in test:
        images.append(images_)
        labels.append(labels_)
    test_images, test_labels = torch.cat(images).numpy(), torch.cat(labels).numpy()

    adversarial_images = torch.load(
        './data/adversarial/untargeted_step10_eps{}.pt'.format(eps)
    )['adversarials']

    projected_images = torch.load(
        './data/projected/untargeted_advstep10_adveps{0}_step10_eps{0}.pt'.format(eps)
    )

    return test_images, test_labels, adversarial_images, projected_images
    

def do_samples(args):

    model = ResNet18().cuda()
    model.load_state_dict(torch.load(args.resnet_fname))
    model.eval()

    test_images, test_labels, adversarial_images, projected_images = \
        get_real_adv_proj(args.eps)

    def get_label(img):
        assert img.shape == (3, 32, 32)
        img = torch.from_numpy(img).unsqueeze(0).cuda()
        return torch.argmax(model(img)).item()

    for i in range(args.n_samples):

        real_image, real_label = test_images[i], test_labels[i]
        real_image = 2.* real_image - 1.
        adversarial_image = adversarial_images[i]
        projected_image = projected_images[i]

        #print('*' * 30)
        #print(real_image[0, :, :])
        #print(adversarial_image[0, :, :])
        #print(projected_image[0, :, :])

        with torch.no_grad():
            adv_img = torch.from_numpy(adversarial_image).unsqueeze(0).cuda()
            adversarial_label = get_label(adversarial_image)
            projected_label = get_label(projected_image)

        print(real_label, adversarial_label, projected_label)

        real_image = np.moveaxis(real_image, 0, -1)
        adversarial_image = np.moveaxis(adversarial_image, 0, -1)
        projected_image = np.moveaxis(projected_image, 0, -1)

        real_image = (real_image + 1.) / 2.
        adversarial_image = (adversarial_image + 1.) / 2.
        projected_image = (projected_image + 1.) / 2.

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.title("Original: {}".format(CIFAR_CLASSES[real_label]))
        plt.imshow(real_image)

        plt.subplot(1, 3, 2)
        plt.title("Adversarial: {}".format(CIFAR_CLASSES[adversarial_label]))
        plt.imshow(adversarial_image)

        plt.subplot(1, 3, 3)
        plt.title("Projected: {}".format(CIFAR_CLASSES[projected_label]))
        plt.imshow(projected_image)

        plt.savefig('figs/samples_eps{}_{}.png'.format(args.eps, i))


def do_histogram(args):

    test_images, _, adversarial_images, projected_images = \
        get_real_adv_proj(args.eps)

    test_images = test_images[:adversarial_images.shape[0]]

    test_images = torch.from_numpy(test_images)
    adversarial_images = torch.from_numpy(adversarial_images)
    projected_images = torch.from_numpy(projected_images)
    test_images = 2. * test_images - 1.

    density = density_generator()

    real_densities = []
    adversarial_densities = []
    projected_densities = []

    for i in range(test_images.shape[0]):

        with torch.no_grad():

            real_densities.append(
                density(test_images[i:i+1].cuda()).item()
            )
            adversarial_densities.append(
                density(adversarial_images[i:i+1].cuda()).item()
            )
            projected_densities.append(
                density(projected_images[i:i+1].cuda()).item()
            )

        if i > 100: break

    print(sum(real_densities) / len(real_densities))
    print(sum(adversarial_densities) / len(adversarial_densities))
    print(sum(projected_densities) / len(projected_densities))
    plt.figure()
    plt.hist(
        [real_densities, adversarial_densities, projected_densities],
        label=['original', 'adversarial', 'projected'],
    )
    #plt.hist(real_densities, label='original', alpha=0.4)
    #plt.hist(adversarial_densities, label='adversarial', alpha=0.4)
    #plt.hist(projected_densities, label='projected', alpha=0.4)
    plt.legend()
    plt.savefig('figs/histogram_eps{}.png'.format(args.eps))


def do_statistics(args):

    model = ResNet18().cuda()
    model.load_state_dict(torch.load(args.resnet_fname))
    model.eval()

    test_images, test_labels, adversarial_images, projected_images = \
        get_real_adv_proj(args.eps)

    test_images = test_images[:adversarial_images.shape[0]]
    test_labels = test_labels[:adversarial_images.shape[0]]

    test_images = torch.from_numpy(test_images).cuda()
    test_labels = torch.from_numpy(test_labels).cuda()
    adversarial_images = torch.from_numpy(adversarial_images).cuda()
    projected_images = torch.from_numpy(projected_images).cuda()
    test_images = 2. * test_images - 1.

    with torch.no_grad():
        test_outputs = torch.argmax(model(test_images), dim=1)
        adversarial_outputs = torch.argmax(model(adversarial_images), dim=1)
        projected_outputs = torch.argmax(model(projected_images), dim=1)

    test_corr = torch.sum(test_outputs == test_labels).item()
    adversarial_corr = torch.sum(adversarial_outputs == test_labels).item()
    projected_corr = torch.sum(projected_outputs == test_labels).item()
    adversarial_diff = torch.sum(adversarial_outputs != test_outputs).item()
    projected_diff = torch.sum(projected_outputs != adversarial_outputs).item()

    n = float(test_images.shape[0])

    print("For eps = {}".format(args.eps))

    print("Original accuracy: {}".format(test_corr / n))
    print("Adversarial accuracy: {}".format(adversarial_corr / n))
    print("Projected accuracy: {}".format(projected_corr / n))
    print("Frac of adv labels different from orig: {}".format(adversarial_diff / n))
    print("Frac of proj labels different form adv: {}".format(projected_diff / n))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--to-show', type=str, required=True)
    parser.add_argument('--resnet-fname', type=str, required=False,
        default='./logdirs/cifar10-40k_best.pt')
    parser.add_argument('--n-samples', type=int, required=False, default=10)
    parser.add_argument('--eps', type=float, required=False, default=0.0001)
    args = parser.parse_args()

    if args.to_show == 'samples':
        do_samples(args)
    elif args.to_show == 'histogram':
        do_histogram(args)
    elif args.to_show == 'statistics':
        do_statistics(args)
    else:
        raise RuntimeError("Invalid argument for 'to_show'")

if __name__ == '__main__':
    main()

