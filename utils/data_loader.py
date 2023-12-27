import h5py
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

class generate_Dataset_h5(Dataset):
    """ImageNet Dataset stored in hdf5 file"""
    def __init__(self, dir_path, transform=False):
        #read the hdf5 file
        self.file = h5py.File(dir_path, 'r')
        self.n_images, self.nx, self.ny, self.nz = self.file['images'].shape
        self.transform = transform

    def __len__(self):
        """number of images in the file"""
        return self.n_images

    def __getitem__(self, idx):
        """return the input image and the associated label"""
        input_h5 = self.file['images'][idx,:,:,:]
        label_h5 = self.file['meta'][idx]
        sample = np.array(input_h5.astype('uint8'))
        label = torch.tensor(int(label_h5))
        sample = self.transform(sample)

        return sample, label


def dataloader(batch_n, scaling_fac=1, data_aug=False, times=1):
    """Return datasets train and valid"""
    # Argument :
    # batch_n : batch_size
    # Num pictures : pictures
    train_path = f"scaling_fac_{scaling_fac}.h5"
    valid_path = "val.h5"
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # ##Training dataset
    train_dataset = generate_Dataset_h5(train_path,
                                        torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(mean=mean, std=std)]))

    if data_aug:
        transforms_list = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.Normalize(mean=mean, std=std)
            ]
        for _ in range(times):
            augmented_dataset = generate_Dataset_h5(train_path, torchvision.transforms.Compose(transforms_list))
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, augmented_dataset])

    # ##Validation dataset
    valid_dataset = generate_Dataset_h5(valid_path,
                                        torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(mean=mean, std=std)]))

    valid_size = len(valid_dataset)
    valid_size = int(valid_size * 0.5)  # Splitting the 'valid' set in half

    valid_dataset, test_dataset = torch.utils.data.random_split(valid_dataset, [valid_size, valid_size])

    train_sampler = None
    valid_sampler = None
    test_sampler = None

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        train_sampler = DistributedSampler(train_dataset)
        valid_sampler = DistributedSampler(valid_dataset)
        test_sampler = DistributedSampler(test_dataset)

    # ##Test dataset
    dataset_loader = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_n, num_workers=0, shuffle=False, sampler=train_sampler, pin_memory=True),
                      'valid': torch.utils.data.DataLoader(valid_dataset, batch_size=batch_n, num_workers=0, shuffle=False, sampler=valid_sampler, pin_memory=True),
                      'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_n, num_workers=0, shuffle=False, sampler=test_sampler, pin_memory=True)
                      }

    dataset_sizes = {'train': len(train_dataset), 'valid' : len(valid_dataset), 'test': len(test_dataset)}

    return dataset_loader, dataset_sizes