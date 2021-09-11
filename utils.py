import torchvision
from torch.utils.data import DataLoader
import torch
import datasets
import transforms


def generate_dataloaders(batch_size, csv, root,name):
    dataset = datasets.VideoDataset(csv,
                                    root,
                                    transform=torchvision.transforms.Compose([transforms.VideoFolderPathToTensor()]))
    torch.save(dataset,name)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True,drop_last=True)

def generate_dataloader(batch_size,name):
    dataset_loaded=torch.load(name)
    return DataLoader(dataset_loaded,
                      batch_size=batch_size,
                      shuffle=True,drop_last=True)

def get_dataloader(batch_size, csv_train, root_train, csv_test, root_test):
    return {
        'train': generate_dataloader(batch_size,"train.pt"),
        'test': generate_dataloader(batch_size,"test.pt")}

def get_dataloaders(batch_size, csv_train, root_train, csv_test, root_test):
    return {
        'train': generate_dataloaders(batch_size, csv_train, root_train,"train.pt"),
        'test': generate_dataloaders(batch_size, csv_test, root_test,"test.pt")}
