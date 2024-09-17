import torch 
import torch.nn as nn
import torch.optim as optim



class net(nn.Module):
    def _init_(self):
        super(net,self)._init_()
        self.flatten=nn.Flatten()
        self.f=nn.Sequential(nn.Linear(28*28,150),nn.Sigmoid(),nn.Linear(150,512),nn.Sigmoid(),nn.Linear(512,10))

    def forward(self,x):
            x=self.flatten(x)
            x=self.f(x)
            return x
menet=net()   