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

# class Down(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel, stride, bn=True):
#         super(Down, self).__init__()
#         self.m = nn.Sequential()
#         self.m.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel, stride))
#         if bn:
#             self.m.add_module('bn', nn.BatchNorm2d(out_channels))
        
#     def forward(self, x):
#         x = self.m(x)
#         return F.leaky_relu(x)
    
    
# class Up(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel, stride, drop=False, bn=True):
#         super(Up, self).__init__()
#         self.m = nn.Sequential()
#         self.m.add_module('conv', nn.ConvTranspose2d(in_channels, out_channels, kernel, stride))
#         if drop:
#             self.m.add_module('drop', nn.Dropout2d(0.4))
#         if bn:
#             self.m.add_module('bn', nn.BatchNorm2d(out_channels))
            
#     def forward(self, x):
#         x = self.m(x)
#         return F.relu(x)
    
    

# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()
#         self.d1 = Down(3, 16, 4, 2, False)
#         self.d2 = Down(16, 32, 3, 2)
#         self.d3 = Down(32, 64, 3, 2) # --- c3
#         self.d4 = Down(64, 128, 3, 2)
#         self.d5 = Down(128, 256, 3, 2) # --- c2
#         self.d6 = Down(256, 512, 3, 2)
#         self.d7 = Down(512, 512, 3, 1, False) 
        
#         self.u1 = Up(512, 512, 3, 1, True) 
#         self.u2 = Up(512, 256, 3, 2) 
#         self.u3 = Up(512, 128, 3, 2) # --- c2
#         self.u4 = Up(128, 64, 3, 2)
#         self.u5 = Up(128, 32, 3, 2) # --- c3
#         self.u6 = Up(32, 16, 3, 2)
#         self.u7 = Up(16, 3, 4, 2, False, False)
        
#     def forward(self, x):
#         c3 = self.d1(x)
#         c3 = self.d2(c3)
#         c3 = self.d3(c3)
#         c2 = self.d4(c3)
#         c2 = self.d5(c2)
#         x = self.d6(c2)
#         x = self.d7(x)
        
#         x = self.u1(x)
#         x = self.u2(x)
#         x = torch.cat([x, c2], dim=1)
#         x = self.u3(x)
#         x = self.u4(x)
#         x = torch.cat([x, c3], dim=1)
#         x = self.u5(x)
#         x = self.u6(x)
#         x = self.u7(x)
#         return torch.sigmoid(x)
    
    
    
if __name__ == '__main__':
    from model import Discriminator, gradient_penalty
    from torch_data import TRAINLOADER
    from time import time
    from torchviz import make_dot
    import pylab
    
    model = UNet()
    disc = Discriminator()
    mae = nn.L1Loss()
    
    since = time()
    
    for k in range(10):
    
        seg, photo = iter(TRAINLOADER).next()
        seg.requires_grad_(True)
        photo.requires_grad_(True)
        
        
        # gen loss: -D(G(z)) + L * MAE
        Gz = model(seg)
        DGz = disc(Gz, seg)
        gen_loss = - DGz.mean() + 100 * mae(Gz, photo)
        
        
        print(gen_loss)
        
        # disc loss: D(G(z)) - D(y) + gp 
        
        # Gz = model(seg)
        # Gz_ = Gz.detach().requires_grad_(True)
        disc_loss = DGz.mean()  - disc(photo, seg).mean() + gradient_penalty(disc, photo, Gz, seg)
        # make_dot(disc_loss).render(filename='graphs/disc')
        # make_dot(gen_loss).render(filename='graphs/gen')
        

        gen_loss.backward(retain_graph=True)
        disc_loss.backward()
        print(disc_loss)
    print((time() - since) / 10)
        