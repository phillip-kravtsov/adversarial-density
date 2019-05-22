import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import foolbox
import foolbox.attacks as attacks

from foolbox.distances import MSE, MAE, Linf, L0
from torch.utils.data import TensorDataset, Subset, ConcatDataset, DataLoader

from train_resnet import get_cifar10_data
from resnet import ResNet18
from pixelcnnpp.density import density_generator

class NoCriteria(foolbox.criteria.Criterion):
    def is_adversarial(predictions, label):
        return False

def get_all_attacks(fmodel):

    ALL_ATTACKS = (
        ('FGSM', attacks.FGSM(fmodel, criterion=NoCriteria, distance=Linf)),
        ('L1', attacks.L1BasicIterativeAttack(fmodel, distance=MAE)),
        ('L2', attacks.L2BasicIterativeAttack(fmodel, distance=MSE)),
        ('CarliniL2Attack', attacks.CarliniWagnerL2Attack(fmodel, distance=MSE)),
    )

    return ALL_ATTACKS

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
                data = linf(data, epsilon/num_update_steps, data_grad)
            elif attack_type == 'l2':
                data = l2(data, epsilon/num_update_steps, data_grad)
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

    fmodel = foolbox.models.PyTorchModel(model, bounds=(-1, 1), num_classes=10)
    _, _, test_loader = get_cifar10_data(bs=1, as_loader=True)

    data = {}

    for name, attack_model in get_all_attacks(fmodel):

        advs = []
        original_classes = []
        print(name)
        for i, (data, target) in enumerate(test_loader):

            if i >= args.num_images:
                break
            data, target = data.cpu().numpy().squeeze(), target.cpu().numpy().squeeze()
            adv = attack_model(data, target, epsilons=(args.epsilon))
            advs.append(adv)

        advs = advs[:args.num_images]
        orig_classes = orig_classes[:args.num_images]
        advs = np.stack(advs)
        orig_classes = np.stack(orig_classes)
        print(advs.shape, orig_classes.shape)

        data[name] = (advs, orig_classes)

    #adversarials, original_classes = get_adversarial_images(
    #    model, torch.device('cuda'), test_loader, args.num_update_steps,
    #    args.num_images, args.epsilon, attack_type=args.attack_type, targeted=False
    #)
    #to_save = {'adversarials': adversarials, 'original_classes': original_classes}
    #torch.save(to_save, args.save_fname)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--epsilon', type=float, required=True)
    parser.add_argument('--num-images', type=int, required=True)
    parser.add_argument('--resnet-fname', type=str, required=False,
        default='./logdirs/cifar10-40k_best.pt')
    parser.add_argument('--attack-type', type=str, required=False, default='linf')
    args = parser.parse_args()

    print(args)
    do_adversarial(args)
        
if __name__ == '__main__':
    main()

