import math
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SRCNN(nn.Module):
    def __init__(self, noi):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(noi, 64, kernel_size=9, padding=2, padding_mode='replicate')
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv2d(32, noi, kernel_size=5, padding=2, padding_mode='replicate')
        self.linear = nn.Linear(2210, 92631)  
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        
        a1, a2, a3, a4 = x.shape
        x2d = torch.reshape(x[:,0,:,:], (a1, a3*a4))
        
        return self.linear(x2d)

    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_data, labels):
        self.image_data = image_data
        self.labels = labels
        
    def __len__(self):
        return (len(self.image_data))
    
    def __getitem__(self, index):
        image = self.image_data[index]
        label = self.labels[index]
        return (
            torch.tensor(image, dtype=torch.float),
            torch.tensor(label, dtype=torch.float)
        )
    
    
def psnr(label, outputs, max_val=1.):
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    img_diff = outputs - label
    rmse = math.sqrt(np.mean((img_diff) ** 2))
    if rmse == 0:
        return 100
    else:
        PSNR = 20 * math.log10(max_val / rmse)
        return PSNR
    
    
def train(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    
    for bi, data in enumerate(dataloader):
        image_data = data[0].to(device)
        label = data[1].to(device)
        
        # zero grad the optimizer
        optimizer.zero_grad()
        outputs = model(image_data)
        out = torch.reshape(outputs, (outputs.shape[0], label.shape[2], label.shape[3]))[:,None,:,:]

        loss = criterion(out, label)
        
        # backpropagation
        loss.backward()
        
        # update the parameters
        optimizer.step()
        
        # add loss of each item (total items in a batch = batch size)
        running_loss += loss.item()
        
        # calculate batch psnr (once every `batch_size` iterations)
        batch_psnr = psnr(label, out)
        running_psnr += batch_psnr
        
    final_loss = running_loss/len(dataloader.dataset)
    final_psnr = running_psnr/int(len(dataloader.dataset)/dataloader.batch_size)
    
    return final_loss, final_psnr


def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    
    with torch.no_grad():
        for bi, data in enumerate(dataloader):
            image_data = data[0].to(device)
            label = data[1].to(device)
            
            outputs = model(image_data)
            out = torch.reshape(outputs, (outputs.shape[0], label.shape[2], label.shape[3]))[:,None,:,:]
            loss = criterion(out, label)
            
            # add loss of each item (total items in a batch = batch size) 
            running_loss += loss.item()
            
            # calculate batch psnr (once every `batch_size` iterations)
            batch_psnr = psnr(label, out)
            running_psnr += batch_psnr
                    
    final_loss = running_loss/len(dataloader.dataset)
    final_psnr = running_psnr/int(len(dataloader.dataset)/dataloader.batch_size)
    
    return final_loss, final_psnr


def run_model(model, train_loader, val_loader, 
              optimizer, criterion, epochs, plot=False,
              outfile='./Data/best_srcnn_model.pt'):
    
    if plot:
        from IPython.display import clear_output
        from collections import defaultdict
        import matplotlib.pyplot as plt
        
        def live_plot(data_dict, figsize=(10,5), title=''):
            clear_output(wait=True)
            plt.figure(figsize=figsize)
            for label,data in data_dict.items():
                plt.plot(data, label=label)
            plt.title(title)
            plt.grid(True)
            plt.legend()
            plt.xlabel('epoch')
            plt.title('PSNR')
            plt.show();
    
    train_loss, val_loss = [], []
    train_psnr, val_psnr = [], []
    psnr_dict = defaultdict(list)
    start = time.time()

    best_psnr = float("-inf")
    for epoch in range(epochs):
        train_epoch_loss, train_epoch_psnr = train(model, train_loader, optimizer, criterion)
        val_epoch_loss, val_epoch_psnr = validate(model, val_loader, criterion)

        if val_epoch_psnr > best_psnr:
            best_psnr = val_epoch_psnr
            torch.save(model.state_dict(), outfile)

        train_loss.append(train_epoch_loss)
        psnr_dict['train_psnr'].append(train_epoch_psnr)
        val_loss.append(val_epoch_loss)
        psnr_dict['val_psnr'].append(val_epoch_psnr)
        
        if plot:
            live_plot(psnr_dict)
            
    end = time.time()
    print(f"Finished training in: {((end-start)/60):.3f} minutes")
    
    return model, psnr_dict, train_loss, val_loss