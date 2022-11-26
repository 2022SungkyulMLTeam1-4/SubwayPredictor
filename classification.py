import torch.nn as nn
from torch.utils.data import Dataset


class SubwayDataset(Dataset):
    def __init__(self):
        super().__init__()


class Model(nn.Module):
    """
    학습용 모델입니다.
    """

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM()

    def forward(self, x):
        x = nn.LSTM
