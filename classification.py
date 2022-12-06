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
        raw_data['최종수신일'] = raw_data['최종수신시간'] // 3600
        raw_data.sort_values(by="최종수신시간", inplace=True)
        group_data = raw_data.groupby(by=['열차번호', '최종수신일'])
        data = [i[1].to_numpy()[:, [0, 2, 3, 4, 5]] for i in group_data]
        data = [i for i in data if i.shape[0] >= 10]

        data = [[self.row_process(row) for row in i] for i in data]
        interpolated_data = [self.interpolation(i) for i in data]

        self.x = []
        self.y = []
        for i in interpolated_data:
            length = len(i) - (len(i) % 10)
            for j in range(0, length, 10):
                self.x.append(i[j: j + 9])
                self.y.append(i[j + 9])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    @staticmethod
    def row_process(row):
        """
        컬럼 데이터를 float형 데이터로 변경하기 위한 메소드입니다.
        """
        distance = abs((row[0] % 1000) - (row[3] % 1000))
        if row[4] == 0:
            distance -= 0.1
        elif row[4] != 1:
            distance += 0.5

        if row[2] == 0:
            distance *= -1

        return np.array([row[1], distance])

    @staticmethod
    def interpolation(data):
        if type(data) == list:
            data = np.array(data)
        min_time = data[:, 0].min()
        max_time = data[:, 0].max()

        interpolated = np.interp(np.arange(min_time, max_time + 1), data[:, 0], data[:, 1])
        x = np.arange(min_time, max_time + 1)

        result = [np.array([x[i], interpolated[i]]) for i in range(x.shape[0])]

        return result


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=4, hidden_size=64, num_layers=2, batch_first=True)
        self.linear = torch.nn.Linear(64, 4)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
