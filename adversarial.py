import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, Subset, ConcatDataset, DataLoader

from train_resnet import get_cifar10_data
from resnet import ResNet18
from pixelcnnpp.density import density_generator

def assert_max_min(tensor, maximum, minimum):
    
    tensor_max, tensor_min = torch.max(tensor), torch.min(tensor)
    middle = (maximum + minimum) / 2.
    
    assert tensor_max <= maximum and tensor_max > middle
    assert tensor_min >= minimum and tensor_min < middle

def fgsm(image, epsilon, data_grad):

    assert_max_min(image, 1., -1.)
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, -1, 1)

    return perturbed_image

def l2_adv(image, epsilon, data_grad):
    
    assert_max_min(image, 1., -1.)
    perturbed_image = image + epsilon * data_grad
    perturbed_image = torch.clamp(perturbed_image, -1, 1)

    return perturbed_image

def get_adversarial_images(model, device, test_loader, num_update_steps, 
        num_images, epsilon, attack_type,targeted=False):

    assert 1 <= num_images <= 10000

    adversarials = []
    original_classes = []

    num_images_completed = 0

    for data, target in test_loader:

        data = 2. * data - 1.
        assert_max_min(data, 1., -1.)
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
            if attack_type == 'fgsm':
                data = fgsm(data, epsilon, data_grad)
            elif attack_type == 'l2':
                data = l2_adv(data, epsilon, data_grad)

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
    model.eval()

    _, _, test_loader = get_cifar10_data(bs=args.bs, as_loader=True)
    adversarials, original_classes = get_adversarial_images(
        model, torch.device('cuda'), test_loader, args.num_update_steps,
        args.num_images, args.epsilon, targeted=False,
    )
    to_save = {'adversarials': adversarials, 'original_classes': original_classes}
    torch.save(to_save, args.save_fname)

def get_projected_images(density, device, loader, num_update_steps, epsilon):

    projections = []

    for data in loader:

        assert_max_min(data, 1., -1.)
        data = data.to(device)
        data.requires_grad = True

        print('Starting new batch of projections ...')

        for _ in range(num_update_steps):
            
            data = torch.cuda.FloatTensor(data.detach())
            data.requires_grad = True
            loss = -torch.mean(density(data))

            print(loss.item())

            data.grad = None
            loss.backward()
            data_grad = data.grad.data
            data = fgsm(data, epsilon, data_grad)

        projections.append(data.cpu().detach().numpy())

    projections = np.concatenate(projections)

    return projections

def do_projection(args):

    adversarials = torch.load(args.adversarials_fname)['adversarials']
    adversarials = torch.from_numpy(adversarials)
    assert_max_min(adversarials, 1., -1.)

    adv_loader = DataLoader(
        adversarials, batch_size=args.bs, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )

    density = density_generator()

    projected = get_projected_images(
        density, torch.device('cuda'), adv_loader, args.num_update_steps, args.epsilon
    )
    torch.save(projected, args.save_fname)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--do-projection', action='store_true')
    parser.add_argument('--save-fname', type=str, required=True)
    parser.add_argument('--epsilon', type=float, required=False)
    parser.add_argument('--num-update-steps', type=int, required=False)
    parser.add_argument('--num-images', type=int, required=False)
    parser.add_argument('--adversarials-fname', type=str, required=False)
    parser.add_argument('--resnet-fname', type=str, required=False,
        default='./logdirs/cifar10-40k_best.pt')
    parser.add_argument('--bs', type=int, required=False, default=256)
    parser.add_argument('--attack-typ', type=str, required=False, default='fgsm')
    args = parser.parse_args()

    print(args)
    if args.do_projection:
        do_projection(args)
    else:
        do_adversarial(args)
        
if __name__ == '__main__':
    main()

