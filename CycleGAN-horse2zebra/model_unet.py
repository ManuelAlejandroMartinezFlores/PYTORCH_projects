from fileinput import filename
import torch
import torch.nn as nn 
import torch.nn.functional as F


# (I + 2P - K) / S + 1
# I + 2 - 3 + 1
class DoubleConv(nn.Module):
    # No change in H, W
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.m = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.m(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.m = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(0.35),
            DoubleConv(in_channels, out_channels)
        )
            
    def forward(self, x):
        return self.m(x)
    
# (I - 1)S - 2P + K     
# 2I - 2 + 2
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.ct = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.drop = nn.Dropout2d(0.3)
        self.dc = DoubleConv(out_channels * 2, out_channels)
        
    def forward(self, x, xcopy):
        x = self.ct(x)
        x = torch.cat([x, xcopy], dim=1)
        x = self.drop(x)
        return self.dc(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.dc = DoubleConv(3, 64)
        self.d1 = Down(64, 128)
        self.d2 = Down(128, 256)
        # self.d3 = Down(256, 512)
        
        # self.u1 = Up(512, 256)
        self.u1 = Up(256, 128)
        self.u2 = Up(128, 64)
        
        self.c = nn.Conv2d(64, 3, 1)
        
    def forward(self, x):
        x1 = self.dc(x)
        x2 = self.d1(x1)
        x = self.d2(x2)
        x = self.u1(x, x2)
        x = self.u2(x, x1)
        x = self.c(x)
        return torch.sigmoid(x)
    
    
if __name__ == '__main__':
    from torch_data import ATEST, BTEST
    from model import Discriminator
    import random 
    from time import time
    from torchviz import make_dot
    
    to_zebra = UNet() # F
    to_horse = UNet() # G
    d_horse = Discriminator() # D
    d_zebra = Discriminator() # D
    
    gan_criteria = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    
    since = time()
    horses, _ = random.choice(ATEST) # X
    zebras, _ = random.choice(BTEST) # Y
    
    horses.requires_grad_(True)
    zebras.requires_grad_(True)
    
    Fx = to_zebra(horses)
    Gy = to_horse(zebras)
    
    GFx = to_horse(Fx)
    FGy = to_zebra(Gy)
    
    DFx = d_zebra(Fx)
    DGy = d_horse(Gy)
    
    Fy = to_zebra(zebras)
    Gx = to_horse(horses)
    
    # gen loss: gan_loss + cycle_loss + identity_loss
    gan_loss_F = gan_criteria(DFx, torch.ones_like(DFx))
    gan_loss_G = gan_criteria(DGy, torch.ones_like(DGy))
    
    cycle_loss = l1_loss(GFx, horses) + l1_loss(FGy, zebras)
    
    identity_loss_F = l1_loss(Fy, zebras) 
    identity_loss_G = l1_loss(Gx, horses)
    
    
    total_loss_F = gan_loss_F + 10 * cycle_loss + 5 * identity_loss_F
    total_loss_G = gan_loss_G + 10 * cycle_loss + 5 * identity_loss_G 
    
    # disc loss: gan_loss 

    Fx_ = Fx.detach().requires_grad_(True)
    Gy_ = Gy.detach().requires_grad_(True)
    DFx = d_zebra(Fx_)
    DGy = d_horse(Gy_)
    
    Dy = d_zebra(zebras)
    Dx = d_horse(horses)
    
    d_zebra_loss = gan_criteria(DFx, torch.zeros_like(DFx)) + gan_criteria(Dy, torch.ones_like(Dy))
    d_horse_loss = gan_criteria(DGy, torch.zeros_like(DGy)) + gan_criteria(Dx, torch.ones_like(Dx))
    
    # make_dot(gan_loss_F).render(filename='graphs/gen')
    # make_dot(cycle_loss).render(filename='graphs/cycle')
    # make_dot(identity_loss_G).render(filename='graphs/identity')
    # make_dot(d_zebra_loss).render(filename='graphs/disc')
    
    
    total_loss_F.backward(retain_graph=True)
    total_loss_G.backward(retain_graph=True)
    d_zebra_loss.backward(retain_graph=True)
    d_horse_loss.backward()
    
    print(time() - since)