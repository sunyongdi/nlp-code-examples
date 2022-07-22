import torch
from torch.utils.data import Dataset
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils import load_pkl


class CustomDataset(Dataset):
    """
    默认使用 List 存储数据
    """
    def __init__(self, fp):
        self.file = load_pkl(fp)
        
    def __getitem__(self, item):
        sample = self.file[item]
        return sample['text_a'], sample['label']

    def __len__(self):
        return len(self.file)