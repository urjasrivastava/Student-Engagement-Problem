from torch.utils.data import Dataset
import pandas as pd
import os
import torch


class VideoDataset(Dataset):

    def __init__(self, csv, root, transform=None):

        self.dataframe = pd.read_csv(csv)
        self.transform = transform
        self.root = root
        

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        label = torch.tensor(self.dataframe.iloc[index][2])
        video = os.path.join(self.root, self.dataframe.iloc[index].ClipID.replace(".avi","").replace(".mp4",""))
        frames = self.transform(video)

        return frames, label
