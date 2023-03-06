import h5py
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset


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

def dataloader(batch_n, num_pictures=1):
    """Return datasets train and valid"""
    # Argument :
    # batch_n : batch_size
    # Num pictures : pictures
    train_path = "%s.h5"%(num_pictures)
    valid_path = "val.h5"
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # ##Training dataset
    train_dataset = generate_Dataset_h5(train_path,
                                        torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(mean=[mean], std=[std])]))

    # ##Validation dataset
    valid_dataset = generate_Dataset_h5(valid_path,
                                        torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(mean=[mean], std=[std])]))

    valid_dataset, test_dataset = torch.utils.data.random_split(valid_dataset, [25000, 25000])

    # ##Test dataset
    dataset_loader = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_n, num_workers=4, shuffle=True),
                      'valid': torch.utils.data.DataLoader(valid_dataset, batch_size=batch_n, num_workers=4, shuffle=True),
                      'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_n, num_workers=4, shuffle=True),

                      }

    dataset_sizes = {'train': len(train_dataset), 'valid' : len(valid_dataset), 'test': len(test_dataset)}

    return dataset_loader, dataset_sizes