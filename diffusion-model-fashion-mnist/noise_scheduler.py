from turtle import forward
import torch 
import torch.nn as nn 
import numpy as np
from data import REAL_LOADER 
import math


def get_betas(steps, min=1e-6, max=1 - 1e-6):
    return torch.linspace(min, max, steps)

def get_alphas(betas):
    alphas = 1 - betas 
    return torch.cumprod(alphas, axis=0)


class Noising(nn.Module):
    def __init__(self, steps=100, min=1e-6, max=1 - 1e-6):
        super(Noising, self).__init__()
        self.alphas = get_alphas(get_betas(steps, min, max)) 
        
        
    def forward(self, x, t):
        alphas = self.alphas[t]
        mean = torch.sqrt(alphas).view(x.size(0), x.size(1), 1, 1) * x 
        std = torch.sqrt(1 - alphas).view(x.size(0), x.size(1), 1, 1)
        noise = torch.randn_like(mean)
        return noise * std + mean, noise
    
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision import transforms as T
    
    N = Noising(6, 1e-6, 0.02)
    
    x, _ = iter(REAL_LOADER).next()
    
    plt.figure()
    t = 0
    for k in range(6):
        y, _ = N(x[0], torch.tensor([t]))
        plt.subplot(2, 3, k+1)
        plt.imshow(T.ToPILImage()(y[0]), cmap='gray')
        plt.axis('off')
        plt.title(f't = {t}')
        t += 1
        
    plt.tight_layout()
    plt.savefig('imgs/noising.png')
        
    
        