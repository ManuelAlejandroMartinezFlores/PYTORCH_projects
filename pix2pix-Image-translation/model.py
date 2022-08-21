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
        self.c1 = nn.Conv2d(6, 64, 4, stride=2)     # (256 - 4)/ 2 + 1 = 127
        self.c2 = nn.Conv2d(64, 128, 3, stride=2)    # (127 - 3)/ 2 + 1 = 63
        self.bn2 = nn.BatchNorm2d(128)
        self.c3 = nn.Conv2d(128, 256, 3, stride=2)   # (63 - 3)/ 2 + 1 =  31
        self.bn3 = nn.BatchNorm2d(256) 
        self.c4 = nn.Conv2d(256, 512, 3, stride=2)   # (31 - 3)/ 2 + 1 = 15
        self.bn4 = nn.BatchNorm2d(512)
        
        self.c = nn.Conv2d(512, 1, 4, stride=1)
        
        
    def forward(self, generated, input):
        x = torch.cat([generated, input], dim=1)
        x = F.leaky_relu(self.c1(x))
        x = self.bn2(self.c2(x))
        x = F.leaky_relu(x)
        x = self.bn3(self.c3(x))
        x = F.leaky_relu(x)
        x = self.c4(x)
        x = F.leaky_relu(self.bn4(x))
        
        return self.c(x)
        
    
    
def gradient_penalty(D, real_samples:torch.Tensor, fake_samples:torch.Tensor, target:torch.Tensor):
    alphas = torch.rand(real_samples.size(0), 1, 1, 1)
    inter = (alphas * real_samples + (1 - alphas) * fake_samples).requires_grad_(True)

    d_inter = D(inter, target)
    fake = torch.autograd.Variable(torch.FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=d_inter,
        inputs=inter,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

if __name__ == '__main__':
    from torch_data import TRAINLOADER
    
    
    seg, photo = iter(TRAINLOADER).next()
    
    model = EncoderDecoder()
    disc = Discriminator()
    
    mae = nn.L1Loss()
    
    # gen loss: -D(G(z)) + L * MAE
    Gz = model(seg)
    
    gen_loss = - disc(Gz, seg).mean() + 100 * mae(Gz, photo)
    gen_loss.backward()
    print(gen_loss)
    
    # disc loss: D(G(z)) - D(y) + gp 
    
    Gz = model(seg)
    disc_loss = disc(Gz, seg).mean() - disc(photo, seg).mean() + gradient_penalty(disc, photo, Gz, seg).mean()
    disc_loss.backward()
    print(disc_loss)
    

    