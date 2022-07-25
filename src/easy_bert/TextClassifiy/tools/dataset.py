import torch
from torch.utils.data import Dataset
from ..utils import load_pkl


def collate_fn(cfg):
    def collate_fn_intra(batch):
        """
    Arg :
        batch () : 数据集
    Returna :
        x (dict) : key为词，value为长度
        y (List) : 关系对应值的集合
    """
        x, y = dict(), []
        input_ids, attention_mask, token_type_ids = [], [], []
        for data, label in batch:
            input_ids.append(data['input_ids'])
            attention_mask.append(data['attention_mask'])
            token_type_ids.append(data['token_type_ids'])
            y.append(int(label))
        x['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
        x['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
        x['token_type_ids'] = torch.tensor(token_type_ids, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.float)
        return x, y

    return collate_fn_intra

class CustomDataset(Dataset):
    """
    默认使用 List 存储数据
    """
    def __init__(self, fp):
        self.file = load_pkl(fp)
        
    def __getitem__(self, item):
        sample = self.file[item]
        return sample['inputs'], sample['label']

    def __len__(self):
        return len(self.file)