from signal import Sigmasks
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.distributions as D
from torch.optim import Adam
from time import time

from model import Model, NLLLoss
from data import get_dataloader 

import seaborn as sns 
import matplotlib.pyplot as plt 


DATA = get_dataloader()

def eval(x, y, mu, sigma):
    x = x.view(-1)
    y = y.view(-1)
    sigma = sigma.view(-1)
    mu = mu.view(-1)
    
    LL1 = mu - sigma 
    UL1 = mu + sigma
    LL2 = mu - 2 * sigma 
    UL2 = mu + 2 * sigma
    LL3 = mu - 3 * sigma 
    UL3 = mu + 3 * sigma
    plt.figure()
    sns.scatterplot(x=x.detach().numpy(), y=y.detach().numpy(), alpha=0.4)
    plt.plot(x.detach().numpy(), mu.detach().numpy(), 'r')
    plt.fill_between(x.detach().numpy(), LL3.detach().numpy(), UL3.detach().numpy(), alpha=0.3)
    plt.fill_between(x.detach().numpy(), LL2.detach().numpy(), UL2.detach().numpy(), alpha=0.2)
    plt.fill_between(x.detach().numpy(), LL1.detach().numpy(), UL1.detach().numpy(), alpha=0.2)
    plt.title('Regression')
    plt.savefig('imgs/test.png')


def train(epochs):
    model = Model()
    optim = Adam(model.parameters(), lr=2e-3)
    
    criteria = NLLLoss()
    
    for epoch in range(epochs):
        since = time()
        loss_ = 0
        for x, y in DATA:
            dist = model(x)
    
            loss = criteria(dist, y)
            
            optim.zero_grad()
            loss.backward() 
            optim.step()
            
            loss_ += loss.item()  / len(DATA)
            
        print(f'epoch: {epoch+1:4d}, loss: {loss_:.4f}, time: {time()-since:.2f}')
            
    dist = model(DATA.dataset.x)

    eval(DATA.dataset.x, DATA.dataset.y, dist.mean, dist.stddev)
    
    
    
    
if __name__ == '__main__':
    train(200)
            
            
            
            

    
    