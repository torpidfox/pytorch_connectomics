import numpy as np

import torch.utils.data
from ..utils import *


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self,
                 datasets: List[torch.utils.data.Dataset],
                 weights: List[float] = [1],
                 proportion: float = 1):
        self.datasets = datasets
        self.weights = weights
        self.proportion = proportion
        
    def __len__(self):
        return max([len(d) for d in self.datasets])
    
    def __getitem__(self, item):
        select_first = np.random.uniform() < self.proportion
        
        if select_first:
            len_first = len(self.datasets[0])
            sample = self.datasets[0][item % len_first]
            return *sample, self.weights[0]
        
        sample = self.datasets[-1][item % len(self.datasets[-1])]
        
        return *sample, self.weights[1]
    
    
    