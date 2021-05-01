from collections import deque

import requests
from torch.utils.data import Dataset


class SimpleAlphaZeroDataset(Dataset):
    def __init__(self, max_length):
        self._memory = deque(maxlen=max_length)

    def push(self, data):
        self._memory.extend(data)

    def __len__(self):
        return len(self._memory)

    def __getitem__(self, i):
        return self._memory[i]


class RemoteDataset:
    def __init__(self, userid, url):
        self._userid = userid
        self._url = url

    def push(self, data):
        data = {'episode': data, 'userid': self._userid}
        requests.post(self._url, json=data)

