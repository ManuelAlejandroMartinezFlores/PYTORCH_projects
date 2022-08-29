import torch
import torch.nn as nn 
import torch.nn.functional as F

# (I + 2P - K) / S + 1
class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()
        self.c1 = nn.Conv2d(in_channels, 16, 4, stride=2)     # (256 - 4)/ 2 + 1 = 127
        self.c2 = nn.Conv2d(16, 64, 3, stride=2)    # (127 - 3)/ 2 + 1 = 63
        self.bn2 = nn.BatchNorm2d(64)
        self.c3 = nn.Conv2d(64, 128, 3, stride=2)   # (63 - 3)/ 2 + 1 =  31
        self.bn3 = nn.BatchNorm2d(128)
        self.c4 = nn.Conv2d(128, 256, 3, stride=2)   # (31 - 3)/ 2 + 1 = 15
        self.bn4 = nn.BatchNorm2d(256)
        self.c5 = nn.Conv2d(256, 512, 3, stride=2)    # (15 - 3)/ 2 + 1 = 7
        self.bn5 = nn.BatchNorm2d(512)
        self.c6 = nn.Conv2d(512, 512, 3, stride=2)    # (7 - 3)/ 2 + 1 = 3
        self.bn6 = nn.BatchNorm2d(512)
        self.c7 = nn.Conv2d(512, 512, 3, stride=1)    # (3 - 3)/ 1 + 1 = 1
        
              
    def forward(self, x):
        x = F.leaky_relu(self.c1(x))
        x = self.bn2(self.c2(x))
        x = F.leaky_relu(x)
        x = self.bn3(self.c3(x))
        x = F.leaky_relu(x)
        x = self.bn4(self.c4(x))
        x = F.leaky_relu(x)
        x = self.bn5(self.c5(x))
        x = F.leaky_relu(x)
        x = self.bn6(self.c6(x))
        x = F.leaky_relu(x)
        return F.leaky_relu(self.c7(x))
        
        
# (I - 1)S - 2P + K 
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.c1 = nn.ConvTranspose2d(512, 512, 3, stride=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.drop1 = nn.Dropout2d(0.4)
        self.c2 = nn.ConvTranspose2d(512, 512, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(512)
        self.drop2 = nn.Dropout2d(0.4)
        self.c3 = nn.ConvTranspose2d(512, 256, 3, stride=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.c4 = nn.ConvTranspose2d(256, 128, 3, stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.c5 = nn.ConvTranspose2d(128, 64, 3, stride=2)
        self.bn5 = nn.BatchNorm2d(64)
        self.c6 = nn.ConvTranspose2d(64, 16, 3, stride=2)
        self.bn6 = nn.BatchNorm2d(16)
        self.c7 = nn.ConvTranspose2d(16, 3, 4, stride=2)
        
        
    def forward(self, x):
        x = self.bn1(self.drop1(self.c1(x)))
        x = F.relu(x)
        x = self.bn2(self.drop1(self.c2(x)))
        x = F.relu(x)
        x = self.bn3(self.c3(x))
        x = F.relu(x)
        x = self.bn4(self.c4(x))
        x = F.relu(x)
        x = self.bn5(self.c5(x))
        x = F.relu(x)
        x = self.bn6(self.c6(x))
        x = F.relu(x)
        x = self.c7(x)
        return torch.sigmoid(x)
    
    
class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
       
        
        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.c1 = nn.Conv2d(3, 64, 4, stride=2)     # (256 - 4)/ 2 + 1 = 127
        self.c2 = nn.Conv2d(64, 128, 3, stride=2)    # (127 - 3)/ 2 + 1 = 63
        self.bn2 = nn.BatchNorm2d(128)
        self.c3 = nn.Conv2d(128, 256, 3, stride=2)   # (63 - 3)/ 2 + 1 =  31
        self.bn3 = nn.BatchNorm2d(256) 
        self.c4 = nn.Conv2d(256, 512, 3, stride=2)   # (31 - 3)/ 2 + 1 = 15
        self.bn4 = nn.BatchNorm2d(512)
        
        self.c = nn.Conv2d(512, 1, 4, stride=1)
        
        
    def forward(self, x):
        x = F.leaky_relu(self.c1(x))
        x = self.bn2(self.c2(x))
        x = F.leaky_relu(x)
        x = self.bn3(self.c3(x))
        x = F.leaky_relu(x)
        x = self.c4(x)
        x = F.leaky_relu(self.bn4(x))
        
        return self.c(x)
    
    
if __name__ == '__main__':
    from torch_data import ATEST, BTEST, get_labels
    import random 
    
    to_zebra = EncoderDecoder() # F
    to_horse = EncoderDecoder() # G
    discriminator = Discriminator() # D
    
    gan_criteria = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    
    horses, _ = random.choice(ATEST) # X
    zebras, _ = random.choice(BTEST) # Y
    
    Fx = to_zebra(horses)
    Gy = to_horse(zebras)
    
    GFx = to_horse(Fx)
    FGy = to_zebra(Gy)
    
    DFx = discriminator(torch.cat([Fx, get_labels(1)], dim=1))
    DGy = discriminator(torch.cat([Gy, get_labels(0)], dim=1))
    
    Dy = discriminator(torch.cat([zebras, get_labels(1)], dim=1))
    Dx = discriminator(torch.cat([horses, get_labels(0)], dim=1))
    
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
    disc_loss = gan_criteria(DFx, torch.zeros_like(DFx)) + gan_criteria(Dy, torch.ones_like(Dy))
    disc_loss += gan_criteria(DGy, torch.zeros_like(DGy)) + gan_criteria(Dx, torch.ones_like(Dx))
    disc_loss *= 0.5 
    
    
    total_loss_F.backward(retain_graph=True)
    total_loss_G.backward(retain_graph=True)
    disc_loss.backward(retain_graph=True)
    
    
    
