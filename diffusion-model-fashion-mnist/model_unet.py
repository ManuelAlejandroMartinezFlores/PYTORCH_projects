import torch
import torch.nn as nn 
import torch.nn.functional as F

from timestep_encoding import SinusoidalPositionEmbedding


# (I + 2P - K) / S + 1
# I + 2 - 3 + 1
class DoubleConv(nn.Module):
    # No change in H, W
    def __init__(self, in_channels, out_channels, time_dim=256):
        super(DoubleConv, self).__init__()
        self.m1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.m2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, out_channels),
            nn.ReLU()
        )
        
    def forward(self, x, t):
        x = self.m1(x)
        t = self.time_mlp(t)[(..., ) + (None, ) * 2]
        x = x + t 
        return self.m2(x)
        


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=256):
        super(Down, self).__init__()
        self.m = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(0.35),
        )
        self.dc = DoubleConv(in_channels, out_channels, time_dim)
            
    def forward(self, x, t):
        x = self.m(x)
        return self.dc(x, t)
    
# (I - 1)S - 2P + K     
# 2I - 2 + 2
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=256):
        super(Up, self).__init__()
        self.ct = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.drop = nn.Dropout2d(0.3)
        self.dc = DoubleConv(out_channels * 2, out_channels, time_dim)
        
    def forward(self, x, xcopy, t):
        x = self.ct(x)
        x = torch.cat([x, xcopy], dim=1)
        x = self.drop(x)
        return self.dc(x, t)
    


class UNet(nn.Module):
    def __init__(self, channels=1, time_dim=256):
        super(UNet, self).__init__()
        self.dc = DoubleConv(channels, 64)
        self.d1 = Down(64, 128)
        self.d2 = Down(128, 256)
        # self.d3 = Down(256, 512)
        
        # self.u1 = Up(512, 256)
        self.u1 = Up(256, 128)
        self.u2 = Up(128, 64)
        
        self.c = nn.Conv2d(64, channels, 1)
        
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
    def forward(self, x, t):
        t = self.time_mlp(t)
        
        x1 = self.dc(x, t)
        x2 = self.d1(x1, t)
        x = self.d2(x2, t)
        x = self.u1(x, x2, t)
        x = self.u2(x, x1, t)
        x = self.c(x)
        return x
    

    
    
if __name__ == '__main__':
    from noise_scheduler import Noising
    from data import REAL_LOADER
    from time import time
    
    model = UNet()
    
    noising = Noising() 
    criteria = nn.MSELoss()
    
    since = time()
    
    imgs, _ = iter(REAL_LOADER).next()
    
    t = torch.rand(imgs.size(0)) * 99 + 1
    t = t.type(torch.long)
    
    noised, noise = noising(imgs, t)
    
    out = model(noised, t)
    loss = criteria(out, noise)
    print(loss)
    loss.backward()
    print(time() - since)
    
    
    
    
    
    