from model import RBFNeuralNet
import torch 
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn as nn 

import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt


df = pd.read_csv('data/banana.csv')

df['Class'] = df['Class'].map(lambda x: (x + 1) // 2)

plt.figure()
sns.scatterplot(x=df['At1'], y=df['At2'], hue=df['Class'], alpha=0.5)
plt.title('banana')
plt.savefig('imgs/data')

X = df[['At1', 'At2']].values 
Y = df['Class'].values 

class Data(Dataset):
    def __init__(self):
        self.x = torch.from_numpy(X).type(torch.float)
        self.y = torch.from_numpy(Y).type(torch.float).view(-1, 1)
        self.n = len(X)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n 
    
DATA = DataLoader(Data(), batch_size=64, shuffle=True)


def train(epochs):
    model = RBFNeuralNet(2, 64)
    optim = Adam(model.parameters(), lr=1e-1)
    criteria = nn.BCELoss()
    
    for epoch in range(epochs):
        eloss = 0
        for x, y in DATA:
            out = model(x)
            loss = criteria(out, y)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            eloss += loss.item()
            
        if epoch % 20 == 19:
            print(f'epoch: {epoch+1:3d}, loss: {eloss/len(DATA):.3f}')
            
            
    x = torch.linspace(-3, 3, 100).repeat(100, 1)
    y = torch.linspace(-3, 3, 100).view(-1, 1).repeat(1, 100)
    z = torch.stack([x, y]).permute((1, 2, 0)).view(-1, 2)
    
    out = model(z)
    
    out = out.view(100, 100, 1).permute((0, 1, 2)).view(100, 100)
    
    plt.figure()
    sns.heatmap(out.detach().numpy(), xticklabels=[], yticklabels=[])
    sns.scatterplot(x=(df['At1']+3)*100/6, y=(df['At2']+3)*100/6, hue=df['Class'], alpha=0.2)
    plt.title('banana')
    plt.savefig('imgs/banana.png')
            
            

if __name__ == '__main__':
    train(200)
    