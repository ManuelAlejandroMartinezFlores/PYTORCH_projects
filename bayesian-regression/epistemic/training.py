import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.distributions as D
from torch.optim import Adam
from time import time

from model import Model
from data import get_dataloader 

import seaborn as sns 
import matplotlib.pyplot as plt 


DATA = get_dataloader()

def eval(x, y, out):
    x = x.view(-1)
    y = y.view(-1)
    mean = torch.cat(out, dim=1).mean(1)
    out = [t.view(-1) for t in out]
    
    

    plt.figure()
    sns.scatterplot(x=x.detach().numpy(), y=y.detach().numpy(), alpha=0.4)
    for l in out:
        plt.plot(x.detach().numpy(), l.detach().numpy(), '--g')
    plt.plot(x.detach().numpy(), mean.detach().numpy(), 'r')
    plt.title('Regression')
    plt.savefig('imgs/test1.png')


def train(epochs):
    model = Model(units=64)
    optim = Adam(model.parameters(), lr=3e-3)
    
    criteria = nn.MSELoss()
    
    for epoch in range(epochs):
        since = time()
        loss_ = 0
        for x, y in DATA:
            out = model(x)
    
            loss = criteria(out, y)
            
            optim.zero_grad()
            loss.backward() 
            optim.step()
            
            loss_ += loss.item()  / len(DATA)
            
        print(f'epoch: {epoch+1:4d}, loss: {loss_:.4f}, time: {time()-since:.2f}')
            
    out = [model(DATA.dataset.x) for _ in range(5)]

    eval(DATA.dataset.x, DATA.dataset.y, out)
    
    
    
    
if __name__ == '__main__':
    train(200)
            
            
            
            

    
    