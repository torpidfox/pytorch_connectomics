import numpy as np

import torch.utils.data
from ..utils import *


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self,
                 datasets: List[torch.utils.data.Dataset],
                 weights: List[float] = [1],
                 proportion: List[float] = 1,
                 mode: str = 'train'):
        self.datasets = datasets
        self.weights = weights
        self.proportion = proportion
        self.volume_size = []
        self.mode = mode
        
        for d in self.datasets:
            self.volume_size += [np.array(x.shape) for x in d.volume]
        
    def __len__(self):
        return max([len(d) for d in self.datasets])
    
    def __getitem__(self, item):
        #fix this

        if self.mode == 'train':
            select_first = np.random.uniform() < self.proportion[0]
        else:
            select_first = True
        
        if select_first:
            len_first = np.floor(len(self.datasets[0]) * self.proportion[0])
            sample = self.datasets[0][item % len_first]
            
            if self.mode == 'train':
                return sample + (self.weights[0],)
            elif self.mode == 'val':
                return sample + (1.0, )
            else:
                return sample
        
        sample = self.datasets[-1][item % len(self.datasets[-1])]
        
        
        if self.mode == 'train':
            return sample + (self.weights[1],)
        
        return sample
    
    
    