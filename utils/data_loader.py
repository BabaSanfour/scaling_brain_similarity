import h5py
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

class generate_Dataset_h5(Dataset):
    """ImageNet Dataset stored in hdf5 file"""

    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path
        self.transform = transform

    def __len__(self):
        """Number of images in the file"""
        with h5py.File(self.dir_path, 'r') as file:
            return file['images'].shape[0]

    def __getitem__(self, idx):
        """Return the input image and the associated label"""
        with h5py.File(self.dir_path, 'r') as file:
            input_h5 = file['images'][idx, :, :, :]
            label_h5 = file['meta'][idx]

        if self.transform:
            sample = self.transform(input_h5.astype('uint8'))

        label = torch.tensor(int(label_h5))


        return sample, label


def dataloader(batch_n, scaling_fac=1, data_aug=False, times=1):
    """Return datasets train and valid"""
    train_path = f"scaling_fac_{scaling_fac}.h5"
    valid_path = "val.h5"
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    transform_list = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ]

    if data_aug:
        transform_list.extend([
            torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
        ])

    transforms = torchvision.transforms.Compose(transform_list)

    # Training dataset
    train_dataset = generate_Dataset_h5(train_path, transform=transforms)

    if data_aug:
        for _ in range(times):
            augmented_dataset = generate_Dataset_h5(train_path, transform=transforms)
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, augmented_dataset])

    # Validation dataset
    valid_dataset = generate_Dataset_h5(valid_path, transform=transforms)

    valid_size = len(valid_dataset)
    valid_size = int(valid_size * 0.5)

    valid_dataset, test_dataset = torch.utils.data.random_split(valid_dataset, [valid_size, valid_size])

    train_sampler = None
    valid_sampler = None
    test_sampler = None

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        train_sampler = DistributedSampler(train_dataset)
        valid_sampler = DistributedSampler(valid_dataset)
        test_sampler = DistributedSampler(test_dataset)
    batch_n = batch_n * 4
    # Test dataset
    dataset_loader = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_n, num_workers=1, shuffle=False,
                                            sampler=train_sampler, pin_memory=True, drop_last=True),
        'valid': torch.utils.data.DataLoader(valid_dataset, batch_size=batch_n, num_workers=1, shuffle=False,
                                            sampler=valid_sampler, pin_memory=True, drop_last=True),
        'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_n, num_workers=1, shuffle=False,
                                           sampler=test_sampler, pin_memory=True, drop_last=True)
    }

    dataset_sizes = {'train': len(train_dataset), 'valid': len(valid_dataset), 'test': len(test_dataset)}

    return dataset_loader, dataset_sizes