import torch
import numpy as np

def get_model_size(model, trainable):
    if trainable:
        # torch. numel ( input ) → int: Returns the total number of elements in the input tensor.
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    else:
        total_params = sum(p.numel() for p in model.parameters())
    return total_params

def seed_experiment(seed):
    """Seed the pseudorandom number generator, for repeatability.
    Args:
        seed (int): random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

def to_device(tensors, device):
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, dict):
        return dict(
            (key, to_device(tensor, device)) for (key, tensor) in tensors.items()
        )
    elif isinstance(tensors, list):
        return list(
            (to_device(tensors[0], device), to_device(tensors[1], device)))
    else:
        raise NotImplementedError("Unknown type {0}".format(type(tensors)))
