import torch

def fgsm(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def get_adversarial_images(model, device, test_loader, epsilon, targeted=False):
    advs = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if targeted:
            raise NotImplementedError
        else:   
            loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = fgsm(data, epsilon, data_grad)
        advs.append(perturbed_data.squeeze().cpu().detach().numpy())
    return advs
        

