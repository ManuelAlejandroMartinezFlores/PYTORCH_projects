import torch.nn as nn 
from torch.optim import Adam 
from torch.optim.lr_scheduler import StepLR
import torch
from time import time

from model import EncoderDecoder, Discriminator, gradient_penalty
from torch_data import TRAINLOADER
from evaluation import visualize_result, evaluation_test


def save(generator, discriminator, gen_optim, disc_optim, epoch):
    data = {
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'gen_optim': gen_optim.state_dict(),
        'disc_optim': disc_optim.state_dict(),
        'epoch': epoch
    }
    torch.save(data, 'models/pix2pix.pth')
    
def load(generator, discriminator, gen_optim, disc_optim):
    try:
        data = torch.load('models/pix2pix.pth')
        generator.load_state_dict(data['generator'])
        discriminator.load_state_dict(data['discriminator'])
        gen_optim.load_state_dict(data['gen_optim'])
        disc_optim.load_state_dict(data['disc_optim'])
        return data['epoch']
    except:
        return 0 
    
    
    
    
def train(epochs, lr=2e-4):
    generator = EncoderDecoder()
    discriminator = Discriminator()
    gen_optim = Adam(generator.parameters(), lr=lr)
    disc_optim = Adam(discriminator.parameters(), lr=lr)
    init_epoch = load(generator, discriminator, gen_optim, disc_optim)
    
    # gen_sch = StepLR(gen_optim, 1, 0.98)
    # disc_sch = StepLR(disc_optim, 1, 0.98)
    
    l1_mae = nn.L1Loss()
    gan_loss = nn.BCEWithLogitsLoss()
    
    for epoch in range(init_epoch, init_epoch + epochs):
        epoch_loss_gen = 0 
        epoch_loss_disc = 0
        epoch_l1_loss = 0
        epoch_gp = 0
        since = time()
        for segmentation, photos in TRAINLOADER:
            
            # gen loss: -D(G(z)) + L * MAE
            Gz = generator(segmentation)
            
            DGz = discriminator(Gz, segmentation)
            gen_loss = gan_loss(DGz, torch.ones_like(DGz))
            
            l1_loss = 100 * l1_mae(Gz, photos)
            
            total_loss = gen_loss + l1_loss
            
            gen_optim.zero_grad()
            total_loss.backward()
            gen_optim.step()
            
            # disc loss: D(G(z)) - D(y) + gp 
            
            Gz = generator(segmentation)
            DGz = discriminator(Gz, segmentation)
            Dx = discriminator(photos, segmentation)
            disc_loss = gan_loss(DGz, torch.zeros_like(DGz)) + gan_loss(Dx, torch.ones_like(Dx))
            # gp = 12 * gradient_penalty(discriminator, photos, Gz, segmentation)
            
            # total_loss = disc_loss + gp 
            total_loss = disc_loss
            
            disc_optim.zero_grad()
            total_loss.backward()
            disc_optim.step()
            
            epoch_loss_disc += disc_loss.item()
            epoch_loss_gen += gen_loss.item()
            epoch_l1_loss += l1_loss.item()
            # epoch_gp += gp.item()
            
            
        
        epoch_loss_disc /= len(TRAINLOADER)
        epoch_l1_loss /= len(TRAINLOADER)
        epoch_gp /= len(TRAINLOADER)
        epoch_loss_gen /= len(TRAINLOADER)
        # disc_sch.step()
        # gen_sch.step()
        
        
        print(f'epoch: {epoch+1:4d}, time: {time()-since:.2f}, gen loss: {epoch_loss_gen:.4f}, disc loss: {epoch_loss_disc:.4f}, l1 loss: {epoch_l1_loss:.4f}, gp: {epoch_gp:.4f}')
        save(generator, discriminator, gen_optim, disc_optim, epoch+1)
        
        if epoch % 10 == 9:
            evaluation_test(f'imgs/epoch{epoch+1:04d}.png')
            
        
        
if __name__ == '__main__':
    train(200)
        
    
    