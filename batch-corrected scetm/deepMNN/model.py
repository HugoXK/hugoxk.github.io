# -*- coding: utf-8 -*-

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

class ResnetBlock(torch.nn.Module):
    """Define a Resnet block"""
    
    def __init__(self, dim):
        """Initialize the Resnet block"""
        
        super(ResnetBlock, self).__init__()

        self.block = self.build_resnet_block(dim)
        
    def build_resnet_block(self, dim):
        block = [torch.nn.Linear(dim, 2*dim),
                 torch.nn.BatchNorm1d(2*dim),
                 torch.nn.PReLU()]

        block += [torch.nn.Linear(2*dim, dim),
                  torch.nn.BatchNorm1d(dim),
                  torch.nn.PReLU()]

        return torch.nn.Sequential(*block)
    
    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.block(x)  # add skip connections
        return out
        

class Net(torch.nn.Module):
    def __init__(self, 
                 input_dim,
                 n_blocks,
                 device):
        super(Net, self).__init__()
        
        model = []
        for i in range(n_blocks):  # add resnet blocks layers
            model += [ResnetBlock(input_dim)]
        self.model = torch.nn.Sequential(*model)
        self.model.to(device=device)
        
    def forward(self, input):
        """Forward function"""
        out = self.model(input)
        return out
    
    
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder  =  nn.Sequential(
            nn.Linear(2000,50),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(50,2000),
            nn.ReLU()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded

class my_dataset(Data.Dataset): 
    def __init__(self,data):
        self.data = data 
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        return self.data[index]
     
def my_train(my_module,my_dataloader):
    criterion = nn.MSELoss()
    device = torch.device("cuda:1")
    optimizer = torch.optim.Adam(my_module.parameters(),lr = 1e-3)
    for epoch in range(100):
        for step,x in enumerate(my_dataloader):
            x = x.to(device=device)
            en_out,de_out = my_module(x)
            loss = criterion(de_out,x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
        print(loss)



