import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from train_resnet import get_cifar10_data
from resnet import ResNet18
from pixelcnnpp.density import density_generator as pixelcnnpp_density_generator

from adversarial_attacks import nothing_attack, linf_gold_attack, l2_gold_attack, deepfool_attack
from adversarial_defenses import nothing_defense, linf_defense, l2_defense, deepfool_defense

np.random.seed(0)
torch.manual_seed(0)

#import sys; sys.path.insert(0, './glow/')
#from glow import density as flow_density

#Use this if you need glowy density
#Usage: bits_per_dim = flow_density_fn(data) where data is numpy arrayand has .shape[0]=size
#flow_density_fn = flow_density.get_bits_fn(batch_size)


ALL_ATTACKS = (
    ('nothing', nothing_attack, {}),
    ('linf-gold-eps0.001-step10', linf_gold_attack, {'epsilon': 0.001, 'num_update_steps': 10}),
    ('linf-gold-eps0.01-step10', linf_gold_attack, {'epsilon': 0.01, 'num_update_steps': 10}),
    ('linf-gold-eps0.1-step10', linf_gold_attack, {'epsilon': 0.1, 'num_update_steps': 10}),
    ('l2-gold-eps0.001-step10', l2_gold_attack, {'epsilon': 0.001, 'num_update_steps': 10}),
    ('l2-gold-eps0.01-step10', l2_gold_attack, {'epsilon': 0.01, 'num_update_steps': 10}),
    ('l2-gold-eps0.1-step10', l2_gold_attack, {'epsilon': 0.1, 'num_update_steps': 10}),
)

ALL_DEFENSES = (
    ('nothing', nothing_defense, {}),
    ('linf-eps0.001-step10', linf_defense, {'epsilon': 0.001, 'num_update_steps': 10}),
    ('linf-eps0.01-step10', linf_defense, {'epsilon': 0.01, 'num_update_steps': 10}),
    ('linf-eps0.1-step10', linf_defense, {'epsilon': 0.1, 'num_update_steps': 10}),
    ('l2-eps0.001-step10', l2_defense, {'epsilon': 0.001, 'num_update_steps': 10}),
    ('l2-eps0.01-step10', l2_defense, {'epsilon': 0.01, 'num_update_steps': 10}),
    ('l2-eps0.1-step10', l2_defense, {'epsilon': 0.1, 'num_update_steps': 10}),
)

def do_adversarial(args):

    assert 1 <= args.num_images <= 10000

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    resnet = ResNet18().to(device)
    resnet.load_state_dict(torch.load(args.resnet_fname))
    resnet.eval()

    _, _, testset = get_cifar10_data(bs=None, as_loader=False)
    testset = torch.utils.data.Subset(testset, list(range(args.num_images)))

    for attack_name, perturb_function, config in ALL_ATTACKS:

        print("Getting adversarials for {} ...".format(attack_name))

        adversarials, gold_classes, original_classes, new_classes = perturb_function(
            testset,
            resnet,
            device,
            config,
        )
        to_save = {
            'adversarials': adversarials,
            'gold': gold_classes,
            'original': original_classes,
            'new': new_classes,
        }
        torch.save(to_save, os.path.join(args.save_dir, 'adversarial/{}.pt'.format(attack_name)))

def do_projection(args):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    density = pixelcnnpp_density_generator()

    for attack_name, _, _ in ALL_ATTACKS:

        adv_data = torch.load(os.path.join(args.save_dir, 'adversarial/{}.pt'.format(attack_name)))
        adversarials = torch.from_numpy(adv_data['adversarials'])

        for defense_name, perturb_function, config in ALL_DEFENSES:

            print("Getting projections for attack = {}, defense = {} ...".format(
                attack_name, defense_name))

            projections = perturb_function(
                adversarials,
                density,
                device,
                config,
            )
            to_save = {
                'projections': projections,
            }
            torch.save(to_save, os.path.join(
                args.save_dir,
                'projected/attack-{}-defense-{}.pt'.format(attack_name, defense_name)
            ))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--do-projection', action='store_true')
    parser.add_argument('--save-dir', type=str, required=False, default='./data')
    parser.add_argument('--resnet-fname', type=str, required=False,
        default='./logdirs/cifar10-40k_best.pt')
    parser.add_argument('--num-images', type=int, required=False, default=1000)
    args = parser.parse_args()

    print(args)
    if args.do_projection:
        do_projection(args)
    else:
        do_adversarial(args)
        
if __name__ == '__main__':
    main()

