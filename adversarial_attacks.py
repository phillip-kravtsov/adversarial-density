import torch
import torch.nn.functional as F
import numpy as np

from adversarial_helpers import assert_max_min, get_dataloader, clean
torch.manual_seed(0)

def nothing_attack(dataset, resnet, device, config):

    adversarials = [clean(tup[0].unsqueeze(0)) for tup in dataset]
    gold_classes = [clean(tup[1].unsqueeze(0)) for tup in dataset]
    original_classes = [
        clean(torch.argmax(resnet(tup[0].unsqueeze(0).to(device)), dim=1)) for tup in dataset
    ]
    new_classes = original_classes

    adversarials = np.concatenate(adversarials)
    gold_classes = np.concatenate(gold_classes)
    original_classes = np.concatenate(original_classes)
    new_classes = np.concatenate(new_classes)

    return adversarials, gold_classes, original_classes, new_classes

def linf_gold_attack(dataset, resnet, device, config):
    '''
    Config requires:
    epsilon - The maximum size of perturbation under linf norm allowed.
    num_update_steps - The number of steps to take.
    '''

    adversarials = []
    gold_classes = []
    original_classes = []
    new_classes = []

    for images, targets in get_dataloader(dataset, bs=64, shuffle=False):

        images, targets = images.to(device).detach(), targets.to(device).detach()
        original_classes.append(clean(torch.argmax(resnet(images), dim=1)))

        for _ in range(config['num_update_steps']):

            images = images.detach()
            images.requires_grad = True
            probs = torch.log_softmax(resnet(images), dim=1)
            loss = F.nll_loss(probs, targets)

            images.grad = None
            loss.backward()
            images = images + config['epsilon'] / config['num_update_steps'] \
                * images.grad.data.sign()

        adversarials.append(clean(images))
        gold_classes.append(clean(targets))
        new_classes.append(clean(torch.argmax(resnet(images), dim=1)))

    adversarials = np.concatenate(adversarials)
    gold_classes = np.concatenate(gold_classes)
    original_classes = np.concatenate(original_classes)
    new_classes = np.concatenate(new_classes)

    return adversarials, gold_classes, original_classes, new_classes


def l2_gold_attack(dataset, resnet, device, config):
    '''
    Config requires:
    epsilon - The maximum size of perturbation under l2 norm allowed.
    num_update_steps - The number of steps to take.
    '''

    def normalize(grad):
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1)
        grad_norm[grad_norm < 1e-15] = 1.
        return grad / grad_norm.reshape(grad.shape[0], 1, 1, 1)

    adversarials = []
    gold_classes = []
    original_classes = []
    new_classes = []

    for images, targets in get_dataloader(dataset, bs=64, shuffle=False):

        images, targets = images.to(device).detach(), targets.to(device).detach()
        original_classes.append(clean(torch.argmax(resnet(images), dim=1)))

        for _ in range(config['num_update_steps']):

            images = images.detach()
            images.requires_grad = True
            probs = torch.log_softmax(resnet(images), dim=1)
            loss = F.nll_loss(probs, targets)

            images.grad = None
            loss.backward()
            images = images + config['epsilon'] / config['num_update_steps'] \
                * normalize(normalize(images.grad.data))

        adversarials.append(clean(images))
        gold_classes.append(clean(targets))
        new_classes.append(clean(torch.argmax(resnet(images), dim=1)))

    adversarials = np.concatenate(adversarials)
    gold_classes = np.concatenate(gold_classes)
    original_classes = np.concatenate(original_classes)
    new_classes = np.concatenate(new_classes)

    return adversarials, gold_classes, original_classes, new_classes


def deepfool_attack(dataset, resnet, device, config):

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

