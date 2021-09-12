import os
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch
from torch import nn, optim

import numpy as np

# from ResTCN import ResTCN
from ResTCN import ResTCN
from utils import get_dataloaders

torch.manual_seed(0)
num_epochs = 30
batch_size = 4
lr = .001
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
print("Device being used:", device, flush=True)


dataloader = get_dataloaders(batch_size,
                            'train.csv',
                            os.path.join(os.getcwd(), 'TrainFrames'),
                            'test.csv',
                            os.path.join(os.getcwd(), 'TestFrames'))
dataset_sizes = {x: len(dataloader[x].dataset) for x in ['train', 'test']}
print(dataset_sizes, flush=True)

model = ResTCN().to(device)
# optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
scheduler = StepLR(optimizer, step_size=50, gamma=.1)

criterion = nn.MSELoss().to(device)
softmax = nn.Softmax()

for epoch in range(num_epochs):

    for phase in ['train', 'test']:

        running_loss = .0
        if phase == 'train':
            model.train()
        else:
            model.eval()
        for inputs, labels in tqdm(dataloader[phase], disable=True):
            inputs = inputs.to(device)
            labels = labels.float().squeeze().to(device)

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        # if phase == 'train':
        #     scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]

        print("[{}] Epoch: {}/{} Loss: {} LR: {}".format(
            phase, epoch + 1, num_epochs, epoch_loss, scheduler.get_last_lr()), flush=True)

torch.save(model,"model.pt")
torch.save(model.state_dict(),"model_state_dict")
