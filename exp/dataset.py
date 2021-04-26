from torch.utils.data import Dataset
from collections import deque


class SimpleAlphaZeroDataset(Dataset):
    def __init__(self, max_length):
        self._memory = deque(maxlen=max_length)

    def push(self, data):
        self._memory.extend(data)

    def __len__(self):
        return len(self._memory)

    def __getitem__(self, i):
        return self._memory[i]
    
