import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import classification

if __name__ == "__main__":
    encoder = classification.Encoder(100, 100, 200, 5, 0.2)
    decoder = classification.Decoder(100, 100, 200, 5, 0.2)
    device = torch.cuda.device(0)
    model = classification.Seq2Seq(encoder, decoder, device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters())

    dataset = classification.SubwayDataset()

    train_dataset, test_dataset = classification.random_split_train_test(dataset, train_ratio=0.8)

    train_loader = DataLoader(train_dataset, batch_size=64)

    for epoch in range(1000000):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            prediction = model(x)
            cost = loss_function(prediction, y)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
