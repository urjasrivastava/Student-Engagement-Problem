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
lr = .0001
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
print("Device being used:", device, flush=True)
dataloader = get_dataloaders(batch_size,
                            'train.csv',
                            'test.csv',)
dataset_sizes = {x: len(dataloader[x].dataset) for x in ['train', 'test']}
print(dataset_sizes, flush=True)
gamma = 2
epsilon=.001
model = ResTCN().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
scheduler = StepLR(optimizer, step_size=50, gamma=.1)
criterion = nn.MSELoss().to(device)

for epoch in range(num_epochs):

    for phase in ['train', 'test']:

        running_loss = .0
        mse_loss =.0
        if phase == 'train':
            model.train()
        else:
            model.eval()
        for inputs, labels in tqdm(dataloader[phase], disable=True):
            inputs = inputs.to(device)
            labels = labels.float().squeeze().to(device)

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs).squeeze()
                mse = criterion(outputs, labels)  
                x = outputs
                y = labels
                vx = x - torch.mean(x)
                vy = y - torch.mean(y)
                cov = torch.pow(torch.dot(vx,vy)/(batch_size),gamma)
                loss = torch.pow(mse,gamma)/(cov + epsilon)
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            mse_loss += mse*inputs.size(0)

        # if phase == 'train':
        #     scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_mse =  mse_loss / dataset_sizes[phase]

        print("[{}] Epoch: {}/{} Loss: {} LR: {} ".format(
            phase, epoch + 1, num_epochs, epoch_mse, scheduler.get_last_lr()), flush=True)
    print("--------------------------------------------------------------------------------")

torch.save(model,"model.pt")
torch.save(model.state_dict(),"model_state_dict")
