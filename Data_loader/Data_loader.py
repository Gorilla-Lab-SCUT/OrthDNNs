import os

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class Data_loader(object):
  def __init__(self, dataset_dir, batch_size = 256, workers = 8, valid_ratio=0, pin_memory=True, shuffle=True):
    error_msg = "[!] valid_ratio should be in the range [0, 1]."
    assert ((valid_ratio >= 0) and (valid_ratio <= 1)), error_msg

    self.train_dir = os.path.join(dataset_dir, 'train')
    self.val_dir = os.path.join(dataset_dir, 'val')

    self.normalize_CIFAR10 = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    
    self.batch_size = batch_size
    self.workers = workers
    self.valid_ratio = valid_ratio
    self.pin_memory = pin_memory
    self.shuffle = shuffle

  def _make_loaders(self, train_dataset, test_dataset):
    if self.valid_ratio > 0 :
        #get split indx
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_ratio * num_train))
        if self.shuffle:
          np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]
        #get new train_val_set
        train_loader = torch.utils.data.DataLoader(
          train_dataset, batch_size=self.batch_size, num_workers=self.workers, pin_memory=self.pin_memory,
          sampler=torch.utils.data.sampler.SubsetRandomSampler(train_idx),
        )

        val_loader = torch.utils.data.DataLoader(
          train_dataset, batch_size=self.batch_size, num_workers=self.workers, pin_memory=self.pin_memory,
          sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_idx),
        )
    else:
        train_loader = torch.utils.data.DataLoader(
              train_dataset, batch_size=self.batch_size, num_workers=self.workers, pin_memory=self.pin_memory,
              shuffle=True,
        )
        val_loader = None
  
    test_loader = torch.utils.data.DataLoader(
      test_dataset, batch_size=self.batch_size, num_workers=self.workers, pin_memory=self.pin_memory,
      shuffle=False,
    )
  
    return {
      "train" : train_loader,
      "test" : val_loader,
      "val" : test_loader,
    }

  def get_CIFAR10_loader(self):
    train_dataset = datasets.CIFAR10(
        root = self.train_dir, train = True, download = True,
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize_CIFAR10
        ]))
    test_dataset = datasets.CIFAR10(
        root=self.val_dir, train=False, download=True,
        transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize_CIFAR10
        ]))

    return self._make_loaders(train_dataset, test_dataset)
