from torch.utils.data import Dataset
import pandas as pd
import os
import torch
from facenet_pytorch import InceptionResnetV1


class VideoDataset(Dataset):

    def __init__(self, csv, root, transform=None):

        self.dataframe = pd.read_csv(csv)
        self.transform = transform
        self.root = root
        self.model_conv = InceptionResnetV1(pretrained='vggface2').eval()
        

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        #label = torch.tensor(self.dataframe.iloc[index][2])
        video = os.path.join(self.root, self.dataframe.iloc[index].ClipID.replace(".avi","").replace(".mp4",""))
        frames = self.transform(video)
        frames = frames.unsqueeze(0);
        z = torch.zeros([4, 512]).cuda()
        for t in range(0,4):
            x = self.model_conv(frames[:,t, :, :, :]) #batchsize*C*H*W
            #x = self.model_linear(x) batchsize*512
            z[t, :] = x #4*512

        return z

class EmbeddingDataset(Dataset):
    
    def __init__(self, csv, train):

        self.dataframe = pd.read_csv(csv)
        self.path=''
        if train:
            self.path = "data_drive_path"
        else :
            self.path = "test_drive_path"     
        

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        label = torch.tensor(self.dataframe.iloc[index][2])
        path = "{}{}".format(self.path,index)
        z = torch.load(path)      

        return z,label
