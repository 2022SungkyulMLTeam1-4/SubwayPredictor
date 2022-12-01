import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import classification

if __name__ == "__main__":
    encoder = classification.Encoder(100, 100, 200, 5, 0.2)
    decoder = classification.Decoder(100, 100, 200, 5, 0.2)
    device = torch.device('cuda:0')
    print(device, type(device))
    model = classification.Seq2Seq(encoder, decoder, device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters())

    dataset = classification.SubwayDataset()

    train_dataset, test_dataset = classification.random_split_train_test(dataset, train_ratio=0.8)

    loader = DataLoader(train_dataset)

    for epoch in range(1000000):
        model.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            prediction = model(x)
            cost = loss_function(prediction, y)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
        
        with model.no_grad():
            model.eval()
            cost = loss_function(*test_dataset[:-1])
            print(f"loss for {epoch} epoch: {cost}")
