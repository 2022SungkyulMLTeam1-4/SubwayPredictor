import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def random_split_train_test(data, train_ratio=0.8):
    train_size = int(train_ratio * len(data))
    test_size = len(data) - train_size
    return torch.utils.data.random_split(data, [train_size, test_size])


class SubwayDataset(Dataset):
    def __init__(self):
        super().__init__()
        raw_data = pd.read_csv('datacsv.csv')
        raw_data = raw_data.drop_duplicates(ignore_index=True)  # 1분마다 수신되는 데이터에 중복 존재함
        raw_data['최종수신시간'] = pd.to_datetime(raw_data['최종수신시간'])
        raw_data['최종수신시간'] = raw_data['최종수신시간'].astype(np.int64) // 10 ** 9 // 60
        data = raw_data.sort_values(by='최종수신시간').groupby(by='열차번호', as_index=False)

        self.data = []
        for i, val in data:
            tensor_value = torch.from_numpy(val.to_numpy())
            length = tensor_value.shape[0]
            padded = torch.nn.functional.pad(tensor_value, (0, 0, 0, 200 - length), 'constant', 0)
            self.data.append(padded)

        self.data = torch.stack(self.data)

        self.x = self.data[:, -1, :]
        self.y = self.data[:, 1:, :]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item], self.y[item]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=4, hidden_size=64, num_layers=2, batch_first=True)
        self.linear = torch.nn.Linear(64, 4)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
