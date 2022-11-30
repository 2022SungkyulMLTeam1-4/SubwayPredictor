import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class SubwayDataset(Dataset):
    def __init__(self):
        super().__init__()
        raw_data = pd.read_csv('datacsv.csv')
        raw_data['최종수신시간'] = pd.to_datetime(raw_data['최종수신시간'])
        raw_data['최종수신시간'] = raw_data['최종수신시간'].astype(int)
        raw_data.drop(columns=['사소한 항목', '전체 지하철 수', 'n번째 지하철', '지하철호선ID', '지하철호선명', '지하철역명', '최종수신날짜', '종착지하철역명'],
                      inplace=True, errors="ignore")
        data = raw_data.sort_values(by='최종수신시간').groupby(by='최종수신시간', as_index=False)

        self.data = []
        for i, val in data:
            self.data.append(torch.from_numpy(val.to_numpy()))

        self.lengths = list(map(len, self.data))
        self.data = torch.nn.utils.rnn.pad_sequence(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def random_split_train_test(data, train_ratio=0.8):
    train_size = int(train_ratio * len(data))
    test_size = len(data) - train_size
    return torch.utils.data.random_split(data, [train_size, test_size])


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))

        outputs, (hidden, cell) = self.lstm(embedded)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_, hidden, cell):
        input_ = input_.unseueeze(0)

        embedded = self.dropout(self.embedding(input_))

        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        prediction = self.fc_out(output.squeeze(0))

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hidden_dim == decoder.hidden_dim, "hidden dimensions of encoder decoder isn't equal"
        assert encoder.n_layers == decoder.n_layers, "layer of encoder decoder isn't equal"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        # 첫 번째 입력값 <sos> 토큰
        input_ = trg[0, :]

        for t in range(0, trg_len):
            output, hidden, cell = self.decoder(input_, hidden, cell)

            # prediction 저장
            outputs[t] = output

            # teacher forcing을 사용할지, 말지 결정
            teacher_force = random.random() < teacher_forcing_ratio

            # 가장 높은 확률을 갖은 값 얻기
            top1 = output.argmax(1)

            # teacher forcing의 경우에 다음 lstm에 target token 입력
            input_ = trg[t] if teacher_force else top1

        return outputs
