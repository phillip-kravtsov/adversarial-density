import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

from torch.utils.data import TensorDataset, Subset, ConcatDataset, DataLoader

from resnet import ResNet18
#from autoaugment import CIFAR10Policy

torch.manual_seed(0)
np.random.seed(0)

def get_cifar10_data(bs, as_loader=True, path='./data'):

    trainset = torchvision.datasets.CIFAR10(
        root=path, train=True, download=True,
        transform=transforms.ToTensor(),
        target_transform=transforms.Lambda(lambda x: torch.tensor(x)),
    )
    testset = torchvision.datasets.CIFAR10(
        root=path, train=False, download=True,
        transform=transforms.ToTensor(),
        target_transform=transforms.Lambda(lambda x: torch.tensor(x)),
    )

    num_train = int(0.8 * len(trainset))
    num_val = len(trainset) - num_train
    trainset, valset = torch.utils.data.random_split(trainset, [num_train, num_val])

    if not as_loader:
        return trainset, valset, testset

    trainloader = DataLoader(
        trainset, batch_size=bs, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False
    )
    valloader = DataLoader(
        valset, batch_size=bs, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False
    )
    testloader = DataLoader(
        testset, batch_size=bs, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False
    )

    return trainloader, valloader, testloader

def get_cifar10_altdata(bs, as_loader=True, path='./data'):

    testdata = np.load('{}/cifar10.1_v6_data.npy'.format(path)) / 256.
    testlabels = np.load('{}/cifar10.1_v6_labels.npy'.format(path))

    testdata = np.moveaxis(testdata, -1, 1)

    if not as_loader:
        return torch.utils.data.TensorDataset(
            torch.from_numpy(testdata), torch.from_numpy(testlabels)
        )

    testloader = DataLoader(
        list(zip(testdata, testlabels)), batch_size=bs, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False
    )

    return testloader

def get_cgan_data(bs, as_loader=True, path='./data/generated'):

    trainset = torch.load('{}/gan_train.pt'.format(path))
    valset   = torch.load('{}/gan_val.pt'.format(path))
    testset  = torch.load('{}/gan_test.pt'.format(path))

    class GANDataset(torch.utils.data.dataset.Dataset):
        def __init__(self, data):
            self.data = data
        def __getitem__(self, index):
            x, y = self.data[index]
            return ((x + 1.) / 2., y)
        def __len__(self):
            return len(self.data)

    trainset = GANDataset(trainset)
    valset = GANDataset(valset)
    testset = GANDataset(testset)

    if not as_loader:
        return trainset, valset, testset

    trainloader = DataLoader(
        trainset, batch_size=bs, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False
    )
    valloader = DataLoader(
        valset, batch_size=bs, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False
    )
    testloader = DataLoader(
        testset, batch_size=bs, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False
    )

    return trainloader, valloader, testloader

def get_biggan_data(bs, trainsize=2000000, as_loader=True, path='./biggan/samples'):

    assert trainsize <= 2000000

    path += '/BigGAN_C10_seed0_Gch64_Dch64_bs50_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema'

    allsamples = np.load('{}/samples.npz'.format(path))
    allx, ally = allsamples['x'], allsamples['y']
    allx = allx[:trainsize + 20000]

    allx, ally = torch.from_numpy(allx).float() / 255., torch.from_numpy(ally)

    trainx, valx, testx = \
        allx[:trainsize], \
        allx[-20000:-10000], \
        allx[-10000:]
    trainy, valy, testy = \
        ally[:trainsize], \
        ally[-20000:-10000], \
        ally[-10000:]

    trainset = TensorDataset(trainx, trainy)
    valset   = TensorDataset(valx, valy)
    testset  = TensorDataset(testx, testy)

    if not as_loader:
        return trainset, valset, testset

    trainloader = DataLoader(
        trainset, batch_size=bs, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False
    )
    valloader = DataLoader(
        valset, batch_size=bs, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False
    )
    testloader = DataLoader(
        testset, batch_size=bs, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False
    )

    return trainloader, valloader, testloader

def get_homogenized_data(bs, as_loader=True, path='./data/homogenized'):

    layer1_samples = np.load('{}/resnet-layer1.samples.npy'.format(path))
    layer1_labels = np.load('{}/resnet-layer1.labels.npy'.format(path))
    layer2_samples = np.load('{}/resnet-layer2.samples.npy'.format(path))
    layer2_labels = np.load('{}/resnet-layer2.labels.npy'.format(path))
    layer3_samples = np.load('{}/resnet-layer3.samples.npy'.format(path))
    layer3_labels = np.load('{}/resnet-layer3.labels.npy'.format(path))

    layer1_samples = (layer1_samples + 1.) / 2.
    layer2_samples = (layer2_samples + 1.) / 2.
    layer3_samples = (layer3_samples + 1.) / 2.

    layer1_samples = torch.from_numpy(layer1_samples)
    layer1_labels = torch.from_numpy(layer1_labels)
    layer2_samples = torch.from_numpy(layer2_samples)
    layer2_labels = torch.from_numpy(layer2_labels)
    layer3_samples = torch.from_numpy(layer3_samples)
    layer3_labels = torch.from_numpy(layer3_labels)

    def get_train_val_test_sets(samples, labels):
        trainset = TensorDataset(samples[:40000], labels[:40000])
        valset   = TensorDataset(samples[40000:50000], labels[40000:50000])
        testset  = TensorDataset(samples[50000:], labels[50000:])
        return trainset, valset, testset

    all_homogenized_data = (
        get_train_val_test_sets(layer1_samples, layer1_labels),
        get_train_val_test_sets(layer2_samples, layer2_labels),
        get_train_val_test_sets(layer3_samples, layer3_labels),
    )

    if not as_loader:
        return all_homogenized_data

    def get_loader(dataset):
        return DataLoader(
            dataset, batch_size=bs, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=False
        )

    all_homogenized_dataloaders = (
        (
            get_loader(all_homogenized_data[0][0]),
            get_loader(all_homogenized_data[0][1]),
            get_loader(all_homogenized_data[0][2]),
        ),
        (
            get_loader(all_homogenized_data[1][0]),
            get_loader(all_homogenized_data[1][1]),
            get_loader(all_homogenized_data[1][2]),
        ),
        (
            get_loader(all_homogenized_data[2][0]),
            get_loader(all_homogenized_data[2][1]),
            get_loader(all_homogenized_data[2][2]),
        ),
    )

    return all_homogenized_dataloaders

class AugmentedDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        itemx, itemy = self.dataset[i]
        itemx = self.transform(itemx)
        return itemx, itemy

def get_all_datasets(data_augment, only_large_gan=False, only_homogenized=False):

    assert not (only_large_gan and only_homogenized), \
        "Cannot return both 'only_large_gan' and 'only_homogenized'."

    cifar10_train, cifar10_val, _ = get_cifar10_data(bs=None, as_loader=False)
    gan_train, gan_val, _ = get_biggan_data(bs=None, as_loader=False)
    resnet_layer1, resnet_layer2, resnet_layer3 = get_homogenized_data(bs=None, as_loader=False)

    resnet_layer1_train, resnet_layer1_val, resnet_layer1_test = resnet_layer1
    resnet_layer2_train, resnet_layer2_val, resnet_layer2_test = resnet_layer2
    resnet_layer3_train, resnet_layer3_val, resnet_layer3_test = resnet_layer3

    if data_augment:
        print('Augmenting data ...')
        transform = transforms.Compose([
            transforms.ToPILImage(),
            CIFAR10Policy(),
            transforms.ToTensor(),
        ])
        cifar10_train = AugmentedDataset(cifar10_train, transform)
        gan_train = AugmentedDataset(gan_train, transform)
        resnet_layer1_train = AugmentedDataset(resnet_layer1_train, transform)
        resnet_layer2_train = AugmentedDataset(resnet_layer2_train, transform)
        resnet_layer3_train = AugmentedDataset(resnet_layer3_train, transform)

    cifar10_10k_gan_30k_train = ConcatDataset(
        [Subset(cifar10_train, range(10000)), Subset(gan_train, range(30000))]
    )
    cifar10_10k_gan_30k_val = ConcatDataset(
        [Subset(cifar10_val, range(2500)), Subset(gan_val, range(7500))]
    )

    cifar10_20k_gan_20k_train = ConcatDataset(
        [Subset(cifar10_train, range(20000)), Subset(gan_train, range(20000))]
    )
    cifar10_20k_gan_20k_val = ConcatDataset(
        [Subset(cifar10_val, range(5000)), Subset(gan_val, range(5000))]
    )

    cifar10_30k_gan_10k_train = ConcatDataset(
        [Subset(cifar10_train, range(30000)), Subset(gan_train, range(10000))]
    )
    cifar10_30k_gan_10k_val = ConcatDataset(
        [Subset(cifar10_val, range(7500)), Subset(gan_val, range(2500))]
    )

    gan_40k_train   = Subset(gan_train, range(40000))
    gan_100k_train  = Subset(gan_train, range(100000))
    gan_500k_train  = Subset(gan_train, range(500000))
    gan_1000k_train = Subset(gan_train, range(1000000))
    gan_1500k_train = Subset(gan_train, range(1500000))
    gan_2000k_train = Subset(gan_train, range(2000000))

    cifar10_40k_gan_40k_train = ConcatDataset(
        [Subset(cifar10_train, range(40000)), Subset(gan_train, range(40000))]
    )
    cifar10_40k_gan_40k_val = ConcatDataset(
        [cifar10_val, gan_val]
    )

    if only_large_gan:
        ALL_SETS = (
            ('gan-2000k', gan_2000k_train, gan_val),
        )

    elif only_homogenized:
        ALL_SETS = (
            ('homogenized-resnet1-40k', resnet_layer1_train, resnet_layer1_val),
            ('homogenized-resnet2-40k', resnet_layer2_train, resnet_layer2_val),
            ('homogenized-resnet3-40k', resnet_layer3_train, resnet_layer3_val),
        )
        
    else:
        ALL_SETS = (
            ('cifar10-40k', cifar10_train, cifar10_val),
            ('cifar10-30k-gan-10k', cifar10_30k_gan_10k_train, cifar10_30k_gan_10k_val),
            ('cifar10-20k-gan-20k', cifar10_20k_gan_20k_train, cifar10_20k_gan_20k_val),
            ('cifar10-10k-gan-30k', cifar10_10k_gan_30k_train, cifar10_10k_gan_30k_val),
            ('gan-40k', gan_40k_train, gan_val),
            ('gan-100k', gan_100k_train, gan_val),
            ('gan-500k', gan_500k_train, gan_val),
            ('gan-1000k', gan_1000k_train, gan_val),
            ('gan-1500k', gan_1500k_train, gan_val),
            ('gan-2000k', gan_2000k_train, gan_val),
            ('cifar10-40k-gan-40k', cifar10_40k_gan_40k_train, cifar10_40k_gan_40k_val),
            ('homogenized-resnet1-40k', resnet_layer1_train, resnet_layer1_val),
            ('homogenized-resnet2-40k', resnet_layer2_train, resnet_layer2_val),
            ('homogenized-resnet3-40k', resnet_layer3_train, resnet_layer3_val),
        )

    return ALL_SETS


def do_eval(model, valloader, criterion):

    model.eval()
    val_losses = []
    val_accs = []

    with torch.no_grad():

        for val_x, val_y in valloader:

            val_x, val_y = val_x.cuda(), val_y.cuda()
            val_x = 2. * val_x - 1

            assert torch.max(val_x) <= 1. and torch.min(val_x) >= -1.

            output = model(val_x)
            val_loss = criterion(output, val_y)
            val_losses.append((val_loss.item(), val_x.shape[0]))

            correct = torch.sum(torch.max(output, dim=1)[1] == val_y)
            acc = correct.item() / float(val_x.shape[0])
            val_accs.append((acc, val_x.shape[0]))

    model.train()

    return val_losses, val_accs

def do_model_training(args, name, trainset, valset):

    trainloader = DataLoader(
        trainset, batch_size=args.bs, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )
    valloader = DataLoader(
        valset, batch_size=args.bs, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )

    model = ResNet18().cuda()

    print('Starting training for {} ...'.format(name))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')

    steps_taken = 0
    train_losses = []

    while True:
        for train_x, train_y in trainloader:

            train_x = 2. * train_x - 1.
            assert torch.max(train_x) <= 1. and torch.min(train_x) >= -1.
            train_x, train_y = train_x.cuda(), train_y.cuda()
            output = model(train_x)
            train_loss = criterion(output, train_y)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            steps_taken += 1

            train_losses.append((train_loss.item(), train_x.shape[0]))

            if steps_taken % args.eval_every == 0:
                
                val_losses, val_accs = do_eval(model, valloader, criterion)

                def get_avg(losses):
                    dot_prod = 0
                    total_size = 0
                    for loss, size in losses:
                        dot_prod += loss * size
                        total_size += size
                    return dot_prod / float(total_size)
                    
                train_loss_avg = get_avg(train_losses)
                val_loss_avg = get_avg(val_losses)
                val_acc_avg = get_avg(val_accs)

                train_losses = []

                print('{0:.3f}, {1:.3f}, {2:.3f}'.format(
                    train_loss_avg, val_loss_avg, val_acc_avg
                ))

                if val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    torch.save(
                        model.state_dict(),
                        args.save_dir + '/{}_best.pt'.format(name)
                    )

            if steps_taken >= args.num_steps:
                return
    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--num-steps', type=int, required=False, default=125000)
    parser.add_argument('--eval-every', type=int, required=False, default=625)
    parser.add_argument('--bs', type=int, required=False, default=64)
    parser.add_argument('--lr', type=float, required=False, default=0.001)
    parser.add_argument('--data-augment', action='store_true', default=False)
    parser.add_argument('--only-homogenized', action='store_true', default=False)
    args = parser.parse_args()

    ALL_SETS = get_all_datasets(args.data_augment, only_homogenized=args.only_homogenized)
    for name, trainset, valset in ALL_SETS:
        do_model_training(args, name, trainset, valset)

if __name__ == '__main__':
    main()

