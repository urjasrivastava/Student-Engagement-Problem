from torch.utils.data import DataLoader
import datasets


def generate_dataloaders(batch_size, csv, train):
    dataset = datasets.EmbeddingDataset(csv,
                                    train)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True,drop_last=True)


def get_dataloaders(batch_size, csv_train, csv_test):
    return {
        'train': generate_dataloaders(batch_size, csv_train,True),
        'test': generate_dataloaders(batch_size, csv_test,False)}
