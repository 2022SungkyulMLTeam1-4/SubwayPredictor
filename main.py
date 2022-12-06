import torch
from torch.utils.data import DataLoader

import classification

if __name__ == "__main__":
    dataset = classification.SubwayDataset()

    train_dataset, test_dataset = classification.random_split_train_test(dataset, train_ratio=0.8)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    model = classification.Model()

    prev_loss, prev_acc = 1e9, 0
    early_stop_counter = 0
    for epoch in range(1000000):
        print(f"epoch {epoch}\n")
        early_stop_counter += 1
        classification.train(train_loader, model)
        loss, acc = classification.test(test_loader, model)
        if loss < prev_loss:
            early_stop_counter = 0
            prev_loss = loss
            torch.save(model, 'model.pth')

        if early_stop_counter >= 5000:
            break

    print("done!")
