from torch.utils.data import DataLoader

import classification

if __name__ == "__main__":
    dataset = classification.SubwayDataset()

    train_dataset, test_dataset = classification.random_split_train_test(dataset, train_ratio=0.8)

    loader = DataLoader(train_dataset)
