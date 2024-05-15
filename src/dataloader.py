from torch.utils.data import Dataset, DataLoader
from src.utils import text_to_inds


class PlaintextDataset(Dataset):
    def __init__(self, filepath):
        with open(filepath, 'r') as f:
            self.text_inds = list(map(lambda text: text_to_inds(text[:-1]), f.readlines()))

    def __len__(self):
        return len(self.text_inds)

    def __getitem__(self, idx):
        return self.text_inds[idx]


class PlaintextDataloader(DataLoader):
    pass
