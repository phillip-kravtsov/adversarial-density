import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from train_resnet import get_cifar10_data
from resnet import ResNet18

def fgsm(image, epsilon, data_grad):

    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image

def get_adversarial_images(model, device, test_loader, num_update_steps, 
        num_images, epsilon, targeted=False):

    assert 1 <= num_images <= 10000

    adversarials = []
    original_classes = []

    num_images_completed = 0

    for data, target in test_loader:

        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        print('Starting new batch of attacks ...')

        for _ in range(num_update_steps):
            
            data = torch.cuda.FloatTensor(data.detach())
            data.requires_grad = True
            output = model(data)
            probs = torch.log_softmax(output, dim=1)

            if targeted:
                raise NotImplementedError
            else:
                loss = F.nll_loss(probs, target)
                print(loss.item())

            data.grad = None
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            data = fgsm(data, epsilon, data_grad)

        adversarials.append(data.squeeze().cpu().detach().numpy())
        original_classes.append(target.cpu().detach().numpy())

        num_images_completed += data.shape[0]
        if num_images_completed >= num_images:
            break

    adversarials = np.concatenate(adversarials)
    original_classes = np.concatenate(original_classes)
    adversarials = adversarials[:num_images]
    original_classes = original_classes[:num_images]

    return adversarials, original_classes

def do_adversarial(args):

    device = torch.device('cuda')

    model = ResNet18().to(device)
    model.load_state_dict(torch.load(args.resnet_fname))

    _, _, test_loader = get_cifar10_data(bs=args.bs, as_loader=True)
    adversarials, original_classes = get_adversarial_images(
        model, torch.device('cuda'), test_loader, args.num_update_steps,
        args.num_images, args.epsilon, targeted=False,
    )
    to_save = {'adversarials': adversarials, 'original_classes': original_classes}
    torch.save(to_save, args.save_fname)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=float, required=True)
    parser.add_argument('--num-update-steps', type=int, required=True)
    parser.add_argument('--num-images', type=int, required=True)
    parser.add_argument('--save-fname', type=str, required=True)
    parser.add_argument('--resnet-fname', type=str, required=False,
        default='./logdirs/cifar10-40k_best.pt')
    parser.add_argument('--bs', type=int, required=False, default=256)
    args = parser.parse_args()

    print(args)
    do_adversarial(args)
        
if __name__ == '__main__':
    main()

