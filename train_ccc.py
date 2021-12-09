from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch
from torch import nn, optim
import numpy as np

# from ResTCN import ResTCN
from ResTCN import ResTCN
from utils import get_dataloaders
from sklearn.metrics import r2_score
torch.manual_seed(0)
num_epochs = 50
batch_size = 512
lr = .0001
use_cuda = True
min_loss = 1e5
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
y_true= torch.tensor([]).to(device)
y_preds = torch.tensor([]).to(device)
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
                if phase == 'test' and epoch == num_epochs-1:
                    y_true =torch.cat((y_true,labels))
                    y_preds= torch.cat((y_preds,outputs))
            
            running_loss += loss.item() * inputs.size(0)
            mse_loss += mse.item()*inputs.size(0)

        # if phase == 'train':
        #     scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_mse =  mse_loss / dataset_sizes[phase]
        if epoch_mse < min_loss and phase =="test":
            min_loss = epoch_mse
        print("[{}] Epoch: {}/{} Loss: {} LR: {} ".format(
            phase, epoch + 1, num_epochs, epoch_mse, scheduler.get_last_lr()), flush=True)
    print("--------------------------------------------------------------------------------")
print("CCC Loss")
y_preds = y_preds.cpu().numpy()
y_true = y_true.cpu().numpy()
print("r2 score on test set : ")
print(r2_score(y_true,y_preds))
print("pearson correlation on test set : ")
print(np.corrcoef(y_true,y_preds)[0][1])
print("Minimum MSE Loss :")
print(min_loss)
torch.save(model,"model.pt")
torch.save(model.state_dict(),"model_state_dict")
