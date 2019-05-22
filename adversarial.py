import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.utils.data import TensorDataset, Subset, ConcatDataset, DataLoader
from torch.autograd.gradcheck import zero_gradients

from train_resnet import get_cifar10_data
from resnet import ResNet18
from pixelcnnpp.density import density_generator
#import sys
#sys.path.insert(0, './glow/')
#from glow import density as flow_density

ALL_ATTACKS = (
    ('linf-eps0.001-step10', linf, {'epsilon': 0.001, 'num_update_steps': 10}),
    ('linf-eps0.01-step10', linf, {'epsilon': 0.01, 'num_update_steps': 10}),
    ('linf-eps0.1-step10', linf, {'epsilon': 0.1, 'num_update_steps': 10}),
    ('l2-eps0.001-step10', l2, {'epsilon': 0.001, 'num_update_steps': 10}),
    ('l2-eps0.01-step10', l2, {'epsilon': 0.01, 'num_update_steps': 10}),
    ('l2-eps0.1-step10', l2, {'epsilon': 0.1, 'num_update_steps': 10}),
)


def assert_is_image(tensor):
    assert tensor.shape == (tensor.shape[0], 3, 32, 32)

def assert_max_min(tensor, maximum, minimum):
    tensor_max, tensor_min = torch.max(tensor), torch.min(tensor)
    middle = (maximum + minimum) / 2.
    assert tensor_max <= maximum and tensor_max > middle
    assert tensor_min >= minimum and tensor_min < middle

def linf(images, targets, model, config):

    for _ in range(config.num_update_steps):

        images = torch.cuda.FloatTensor(images.detach())
        images.requires_grad = True
        output = model(images)
        probs = torch.log_softmax(output, dim=1)
        loss = F.nn_loss(probs, targets)

        images.grad = None
        loss.backward()
        images = images + config.epsilon * images.grad.data.sign()

    return images


def l2(images, targets, model, config):

    def normalize(grad):
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1)
        grad_norm[grad_norm < 1e-15] = 1.
        return grad / grad_norm.reshape(grad.shape[0], 1, 1, 1)

    for _ in range(config.num_update_steps):

        images = torch.cuda.FloatTensor(images.detach())
        images.requires_grad = True
        output = model(images)
        probs = torch.log_softmax(output, dim=1)
        loss = F.nn_loss(probs, targets)

        images.grad = None
        loss.backward()
        images = images + config.epsilon * normalize(normalize(images.grad.data))

    return images


def deepfool(images, targets, model, config):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")


    f_image = net.forward(image[None, :, :, :]).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = pert_image[None, :]
    x.requires_grad = True
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = pert_image
        x.requires_grad = True
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, label, k_i, pert_image

def get_adversarial_images(model, device, test_loader, num_update_steps, 
        num_images, epsilon, attack_type, targeted=False):

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

        if attack_type == 'deepfool':
            _, _, _, _, data = deepfool(data.squeeze().detach(), model, num_update_steps)
            data = data.unsqueeze(0)

        else:
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
                if attack_type == 'linf':
                    data = linf(data, epsilon, data_grad)
                elif attack_type == 'l2':
                    data = l2(data, epsilon, data_grad)
                else:
                    raise NotImplementedError

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
        model, torch.device('cuda'), test_loader, args.num_update_steps, args.num_images, 
        args.epsilon / args.num_update_steps if args.epsilon is not None else None,
        attack_type=args.attack_type, targeted=False
    )
    to_save = {'adversarials': adversarials, 'original_classes': original_classes}
    torch.save(to_save, args.save_fname)

def get_projected_images(density, device, loader, num_update_steps, epsilon):

    projections = []

    #Use this if you need glowy density
    #Usage: bits_per_dim = flow_density_fn(data) where data is numpy arrayand has .shape[0]=size
    #flow_density_fn = flow_density.get_bits_fn(batch_size)

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
            data = linf(data, epsilon, data_grad)

        projections.append(data.cpu().detach().numpy())

    projections = np.concatenate(projections)

    return projections

def do_projection(args):

    adversarials = torch.load(args.adversarials_fname)['adversarials']
    adversarials = torch.from_numpy(adversarials)
    adversarials = torch.clamp(adversarials, max=1., min=-1.)
    assert_max_min(adversarials, 1., -1.)

    if len(adversarials.shape) == 3:
        assert adversarials.shape == (1000, 32, 32)
        adversarials = adversarials[:999, :, :].reshape(333, 3, 32, 32)

    adv_loader = DataLoader(
        adversarials, batch_size=args.bs, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )

    density = density_generator()

    projected = get_projected_images(
        density, torch.device('cuda'), adv_loader, args.num_update_steps,
        args.epsilon / args.num_update_steps,
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
    parser.add_argument('--attack-type', type=str, required=False, default='linf')
    args = parser.parse_args()

    print(args)
    if args.do_projection:
        do_projection(args)
    else:
        do_adversarial(args)
        
if __name__ == '__main__':
    main()

