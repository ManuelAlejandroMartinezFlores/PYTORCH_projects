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
        
        self.y_mlp = nn.Sequential(
            nn.Linear(time_dim, out_channels),
            nn.ReLU()
        )
        
    def forward(self, x, t, y, use_label):
        x = self.m1(x)
        t = self.time_mlp(t)[(..., ) + (None, ) * 2]
        y = (self.y_mlp(y) * use_label.view(-1, 1))[(..., ) + (None, ) * 2]
        x = x + t + y
        return self.m2(x)
        


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=256):
        super(Down, self).__init__()
        self.m = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(0.35),
        )
        self.dc = DoubleConv(in_channels, out_channels, time_dim)
            
    def forward(self, x, t, y, use_label):
        x = self.m(x)
        return self.dc(x, t, y, use_label)
    
# (I - 1)S - 2P + K     
# 2I - 2 + 2
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=256):
        super(Up, self).__init__()
        self.ct = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.drop = nn.Dropout2d(0.3)
        self.dc = DoubleConv(out_channels * 2, out_channels, time_dim)
        
    def forward(self, x, xcopy, t, y, use_label):
        x = self.ct(x)
        x = torch.cat([x, xcopy], dim=1)
        x = self.drop(x)
        return self.dc(x, t, y, use_label)
    


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_dim=256):
        super(UNet, self).__init__()
        self.dc = DoubleConv(in_channels, 64)
        self.d1 = Down(64, 128)
        self.d2 = Down(128, 256)
        # self.d3 = Down(256, 512)
        
        # self.u1 = Up(512, 256)
        self.u1 = Up(256, 128)
        self.u2 = Up(128, 64)
        
        self.c = nn.Conv2d(64, out_channels, 1)
        
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
        self.y_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
    def forward(self, x, t, labels=None, use_label=None):
        if labels is None:
            labels = torch.zeros(x.size(0))
        if use_label is None:
            use_label = torch.zeros(x.size(0))
        
        t = self.time_mlp(t)
        y = self.y_mlp(labels)
        
        x1 = self.dc(x, t, y, use_label)
        x2 = self.d1(x1, t, y, use_label)
        x = self.d2(x2, t, y, use_label)
        x = self.u1(x, x2, t, y, use_label)
        x = self.u2(x, x1, t, y, use_label)
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
    
    imgs, labels = iter(REAL_LOADER).next()
    
    t = torch.randint(0, 100, (imgs.size(0), )).type(torch.long)
    use_label = torch.randint(0, 2, (imgs.size(0), )).type(torch.long)
    t = t.type(torch.long)
    
    noised, noise = noising(imgs, t)

    out = model(noised, t, labels, use_label)
    loss = criteria(out, noise)
    print(loss)
    loss.backward()
    print(time() - since)
    
    
    
    
    
    