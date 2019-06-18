import torch

def assert_max_min(tensor, maximum, minimum):
    tensor_max, tensor_min = torch.max(tensor), torch.min(tensor)
    middle = (maximum + minimum) / 2.
    assert tensor_max <= maximum and tensor_max > middle
    assert tensor_min >= minimum and tensor_min < middle

def get_dataloader(dataset, bs, shuffle=True):

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
    )

    return dataloader

def clean(tensor):
    return tensor.detach().cpu().numpy()

