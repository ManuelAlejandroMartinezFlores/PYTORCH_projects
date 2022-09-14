import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim import Adam
from time import time

from model_unet import UNet
from sampling import Sampler
from data import REAL_LOADER 
from noise_scheduler import Noising


def load(model:UNet, optim:Adam):
    try:
        data = torch.load('models/diffusion-mnist.pth')
        model.load_state_dict(data['model'])
        optim.load_state_dict(data['optim'])
        print('loaded model')
        return data['epoch']
    except:
        return 0 
    
    
def save(model:UNet, optim:Adam, epoch:int):
    data = {
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'epoch': epoch
    }
    torch.save(data, 'models/diffusion-mnist.pth')
    
    
    
def training(epochs):
    model = UNet()
    optim = Adam(model.parameters(), lr=1e-3)
    iepoch = load(model, optim)
    criteria = nn.MSELoss()
    
    noising = Noising(max=0.02)
    sampler = Sampler(model, max=0.02)
    
    for epoch in range(iepoch, iepoch + epochs):
        since = time()
        for imgs, _ in REAL_LOADER:
            t = torch.randint(0, 100, (imgs.size(0), )).type(torch.long)
            
            noised, noise = noising(imgs, t)
            
            out = model(noised, t)
            loss = criteria(out, noise)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
        print(f'epoch: {epoch+1:4d}, loss: {loss.item():.4f}, time: {time() - since:.0f}')
        save(model, optim , epoch+1)
        
        if epoch % 5 == 4:
            sampler.plot_samples(f'imgs/epoch{epoch+1:04d}.png')
        
        
if __name__ == '__main__':
    print(len(REAL_LOADER))
    training(500)