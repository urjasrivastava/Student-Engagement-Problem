from torch.utils import data
import datasets
import pandas as pd
import os
import torchvision
import transforms
import torch

dataset = datasets.VideoDataset('train.csv',
                            os.path.join(os.getcwd(), 'TrainFrames'),
                            transform=torchvision.transforms.Compose([transforms.VideoFolderPathToTensor()]))

for idx, data in enumerate(dataset):
        torch.save(data, 'data_drive_path{}'.format(idx))


dataset = datasets.VideoDataset('test.csv',
                            os.path.join(os.getcwd(), 'TestFrames'),
                            transform=torchvision.transforms.Compose([transforms.VideoFolderPathToTensor()]))
                            
for idx, data in enumerate(dataset):
        torch.save(data, 'test_drive_path{}'.format(idx))
