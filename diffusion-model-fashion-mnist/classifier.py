import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim import Adam
from time import time 

from data import REAL_LOADER
from noise_scheduler import Noising
from timestep_encoding import SinusoidalPositionEmbedding


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        
        self.c1 = nn.Conv2d(1, 32, 2, stride=2)     # (28 - 2)/ 2 + 1 = 14
        self.t1 = nn.Linear(256, 32)
        self.c2 = nn.Conv2d(32, 64, 2, stride=2)    # (14 - 2)/ 2 + 1 = 7
        self.bn2 = nn.BatchNorm2d(64)
        self.t2 = nn.Linear(256, 64)
        self.c3 = nn.Conv2d(64, 128, 3, stride=2)   # (7 - 3)/ 1 + 1 =  3
        self.bn3 = nn.BatchNorm2d(128) 
        self.t3 = nn.Linear(256, 128)
        
        self.c = nn.Conv2d(128, 256, 3, stride=1)     # (3 - 3)/ 1 + 1 = 1
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(256 * 2, 256)
        self.fc2 = nn.Linear(256, 10)
        
        
        
    def forward(self, x, t):
        t = self.time_mlp(t) 
        x = F.leaky_relu(self.c1(x))
        t1 = self.t1(t).view(x.size(0), 32, 1, 1)
        x = x + t1
        x = self.bn2(self.c2(x))
        x = F.leaky_relu(x)
        t2 = self.t2(t).view(x.size(0), 64, 1, 1)
        x = x + t2 
        x = self.bn3(self.c3(x))
        x = F.leaky_relu(x)
        t3 = self.t3(t).view(x.size(0), 128, 1, 1)
        x = x + t3
        
        x = self.c(x)
        x = F.leaky_relu(x)
        x = self.flatten(x)
        x = torch.cat([x, t], dim=1)
        
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)
    
    
def save(model:Discriminator, optim:Adam, epoch:int):
    data = {
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'epoch': epoch,
    }
    torch.save(data, 'models/classifier.pth')
    
def load(model:Discriminator, optim:Adam):
    try:
        data = torch.load('models/classifier.pth')
        model.load_state_dict(data['model'])
        optim.load_state_dict(data['optim'])
        print('model loaded')
        return data['epoch']
    except:
        return 0
    
def train(epochs):
    model = Discriminator()
    optim = Adam(model.parameters(), lr=1e-3)
    initepoch = load(model, optim)
    
    cirteria = nn.CrossEntropyLoss()
    noising = Noising(max=0.02)
    
    for epoch in range(initepoch, initepoch + epochs):
        eloss = 0
        since = time()
        for imgs, labels in REAL_LOADER:
            
            t = torch.randint(0, 100, (imgs.size(0), )).type(torch.long)
            
            noised, _ = noising(imgs, t)
            
            out = model(noised, t)
            
            loss = cirteria(out, labels)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            eloss += loss.item()
            
        eloss /= len(REAL_LOADER)
            
        print(f'epoch: {epoch+1:3d}, loss: {eloss:.4f}, time: {time() - since:.0f}')
        save(model, optim, epoch+1)
            
        
            
    
    
    
    
if __name__ == '__main__':
    
    train(10)