import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class SubwayDataset(Dataset):
    def __init__(self):
        super().__init__()
        raw_data = pd.read_csv('datacsv.csv')
        raw_data['최종수신시간'] = pd.to_datetime(raw_data['최종수신시간'])
        raw_data.drop(columns=['사소한 항목', '전체 지하철 수', 'n번째 지하철', '지하철호선ID', '지하철호선명', '지하철역명', '최종수신날짜', '종착지하철역명'],
                      inplace=True, errors="ignore")
        data = raw_data.sort_values(by='최종수신시간').groupby(by='최종수신시간', as_index=False)

        self.data = []
        for i, val in data:
            self.data.append(torch.tensor(val.values))

        lengths = list(map(len, self.data))
        self.data = torch.nn.utils.rnn.pad_sequence(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        pass


class Model(nn.Module):
    """
    학습용 모델입니다.
    """

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(100, 200, 5, batch_first=True)

    def forward(self, x):
        x = self.lstm(x)
