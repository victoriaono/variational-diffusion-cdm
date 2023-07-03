''' 
Augmented dataset
'''
from torch.utils.data import Dataset

class AugmentedDataset(Dataset):

    def __init__(self, *data, transform=None):
        assert all(data[0].size(0) == tensor.size(0) for tensor in data)

        self.tensors = data
        self.transform = transform

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]

        if self.transform:
            x, y = self.transform((x, y))
        
        return x, y