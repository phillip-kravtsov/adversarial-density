import torch
import torch.nn.functional as F
import numpy as np

from adversarial_helpers import assert_max_min, get_dataloader, clean
torch.manual_seed(0)

def nothing_defense(dataset, density, device, config):
    return clean(dataset)

def linf_defense(dataset, density, device, config):
    '''
    Config requires:
    epsilon - The maximum size of perturbation under linf norm allowed.
    num_update_steps - The number of steps to take.
    '''

    projections = []

    for images in get_dataloader(dataset, bs=10, shuffle=True):

        images = images.to(device).detach()

        for _ in range(config['num_update_steps']):

            images = images.detach()
            images.requires_grad = True
            loss = -torch.sum(density(images))

            images.grad = None
            loss.backward()
            images = images + config['epsilon'] / config['num_update_steps'] \
                * images.grad.data.sign()

        projections.append(clean(images))

    projections = np.concatenate(projections)

    return projections


def l2_defense(dataset, density, device, config):
    '''
    Config requires:
    epsilon - The maximum size of perturbation under l2 norm allowed.
    num_update_steps - The number of steps to take.
    '''

    def normalize(grad):
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1)
        grad_norm[grad_norm < 1e-15] = 1.
        return grad / grad_norm.reshape(grad.shape[0], 1, 1, 1)

    projections = []

    for images in get_dataloader(dataset, bs=10, shuffle=True):

        images = images.to(device).detach()

        for _ in range(config['num_update_steps']):

            images = images.detach()
            images.requires_grad = True
            loss = -torch.sum(density(images))

            images.grad = None
            loss.backward()
            images = images + config['epsilon'] / config['num_update_steps'] \
                * normalize(normalize(images.grad.data))

        projections.append(clean(images))

    projections = np.concatenate(projections)

    return projections

def deepfool_defense(dataset, density, device, config):
    raise NotImplementedError()

