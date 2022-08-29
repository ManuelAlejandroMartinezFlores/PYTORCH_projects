import torch.nn as nn 
from torch.optim import Adam 
import torch
from time import time
import random
from torch.optim.lr_scheduler import StepLR

from model import EncoderDecoder, Discriminator
from model_unet import UNet
from torch_data import ATRAIN, BTRAIN, get_labels
from evaluation import evaluate



def load(to_horse:nn.Module, to_zebra:nn.Module, D_horse:nn.Module, D_zebra:nn.Module,
         horse_optim:Adam, zebra_optim:Adam, d_horse_optim:Adam, d_zebra_optim:Adam):
    try:
        data = torch.load('models/cycle-gan.pth')
        to_horse.load_state_dict(data['to_horse'])
        to_zebra.load_state_dict(data['to_zebra'])
        D_horse.load_state_dict(data['d_horse'])
        D_zebra.load_state_dict(data['d_zebra'])
        horse_optim.load_state_dict(data['horse_optim'])
        zebra_optim.load_state_dict(data['zebra_optim'])
        d_horse_optim.load_state_dict(data['d_horse_optim'])
        d_zebra_optim.load_state_dict(data['d_zebra_optim'])
        return data['epoch']
    except:
        return 0 
    
    
def save(to_horse:nn.Module, to_zebra:nn.Module, D_horse:nn.Module, D_zebra:nn.Module,
         horse_optim:Adam, zebra_optim:Adam, d_horse_optim:Adam, d_zebra_optim:Adam, epoch:int):
    data = {
        'to_horse': to_horse.state_dict(),
        'to_zebra': to_zebra.state_dict(),
        'd_horse': D_horse.state_dict(),
        'd_zebra': D_zebra.state_dict(),
        'horse_optim': horse_optim.state_dict(),
        'zebra_optim': zebra_optim.state_dict(),
        'd_horse_optim': d_horse_optim.state_dict(),
        'd_zebra_optim': d_zebra_optim.state_dict(),
        'epoch': epoch
    }
    torch.save(data, 'models/cycle-gan.pth')
    

def train(epochs): 
    to_horse = UNet() # G
    to_zebra = UNet() # F
    D_horse = Discriminator()
    D_zebra = Discriminator()
    
    horse_optim = Adam(to_horse.parameters(), lr=2e-4)
    zebra_optim = Adam(to_zebra.parameters(), lr=2e-4)
    d_horse_optim = Adam(D_horse.parameters(), lr=2e-4)
    d_zebra_optim = Adam(D_zebra.parameters(), lr=2e-4)
    
    init_epoch = load(to_horse, to_zebra, D_horse, D_zebra,
                        horse_optim, zebra_optim, d_horse_optim, d_zebra_optim)
    
    h_s = StepLR(horse_optim, 1, 2)
    z_s = StepLR(zebra_optim, 1, 2)
    dh_s = StepLR(d_horse_optim, 1, 2)
    dz_s = StepLR(d_zebra_optim, 1, 2)
    
    gan_criteria = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    
    for epoch in range(init_epoch, init_epoch + epochs):
        if epoch == 130:
            h_s.step()
            z_s.step()
            dh_s.step()
            dz_s.step()
            print('step')
        
        
        since = time()
        F_epoch_loss = 0
        G_epoch_loss = 0
        Dx_epoch_loss = 0
        Dy_epoch_loss = 0
        I_epoch_loss = 0
        C_epoch_loss = 0
        for iter in range(100):
            horses, _ = random.choice(ATRAIN)
            zebras, _ = random.choice(BTRAIN)
            
            if horses.size(1) != 3 or zebras.size(1) != 3: continue
            
            Fx = to_zebra(horses)
            Gy = to_horse(zebras)
            
            GFx = to_horse(Fx)
            FGy = to_zebra(Gy)
            
            DFx = D_zebra(Fx)
            DGy = D_horse(Gy)
            
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
            DFx = D_zebra(Fx_)
            DGy = D_horse(Gy_)
            Dy = D_zebra(zebras)
            Dx = D_horse(horses)
            
            d_zebra_loss = gan_criteria(DFx, torch.zeros_like(DFx)) + gan_criteria(Dy, torch.ones_like(Dy))
            d_horse_loss = gan_criteria(DGy, torch.zeros_like(DGy)) + gan_criteria(Dx, torch.ones_like(Dx))
            
            
            horse_optim.zero_grad()
            zebra_optim.zero_grad()
            d_horse_optim.zero_grad()
            d_zebra_optim.zero_grad()
            total_loss_F.backward(retain_graph=True)
            total_loss_G.backward(retain_graph=True)
            d_zebra_loss.backward()
            d_horse_loss.backward()
            horse_optim.step()
            zebra_optim.step()
            d_horse_optim.step()
            d_zebra_optim.step()
            
            F_epoch_loss += gan_loss_F.item()
            G_epoch_loss += gan_loss_G.item()
            Dx_epoch_loss += d_horse_loss.item()
            Dy_epoch_loss += d_zebra_loss.item()
            I_epoch_loss += identity_loss_F.item() + identity_loss_G.item()
            C_epoch_loss += cycle_loss.item()
            
            
            
        F_epoch_loss /= 100
        G_epoch_loss /= 100
        Dx_epoch_loss /= 100
        Dy_epoch_loss /= 100
        I_epoch_loss /= 100
        C_epoch_loss /= 100
            
            
        txt = f'Epoch: {epoch+1:4d}, F loss: {F_epoch_loss:.3f}, G loss: {G_epoch_loss:.3f}, Dx loss: {Dx_epoch_loss:.3f}, Dy loss: {Dy_epoch_loss:.3f},\n'
        print(txt + f'\tI loss: {I_epoch_loss:.4f}, C loss: {C_epoch_loss:.4f}, time {time()-since:.2f}')
        save(to_horse, to_zebra, D_horse, D_zebra, horse_optim, zebra_optim, d_horse_optim, d_zebra_optim, epoch+1)
        
        if epoch % 10 == 9:
            evaluate(f'imgs/epoch{epoch+1:04d}.png')


if __name__ == '__main__':
    train(500)        
        