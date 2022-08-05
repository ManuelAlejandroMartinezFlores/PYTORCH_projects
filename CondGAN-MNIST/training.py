import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
from time import time

from model import CondCriticMNIST, CondGeneratorMNIST

batch_size = 32
input_size = 100
lr = 1e-4

generator = CondGeneratorMNIST()
critic = CondCriticMNIST()
gen_opt = optim.Adam(generator.parameters(), lr=lr)
crit_opt = optim.Adam(critic.parameters(), lr=lr)
gen_sch = lr_scheduler.StepLR(gen_opt, step_size=1, gamma=0.95)
crit_sch = lr_scheduler.StepLR(crit_opt, step_size=1, gamma=0.95)

train_dataset = datasets.MNIST(root="./data", train=True,
                                           transform=transforms.ToTensor())
REAL_LOADER = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True)


def random_batch(batch_size, input_size, labels):
    y = F.one_hot(labels, num_classes=10).view(-1, 10, 1, 1)
    z = torch.randn(batch_size, input_size, 1, 1)
    return torch.cat([z, y], dim=1), y

    

def load_model(filename):
    try:
        data = torch.load(f'{filename}/CondGAN.pth')
        generator.load_state_dict(data['generator'])
        critic.load_state_dict(data['critic'])
        gen_opt.load_state_dict(data['gen_optim'])
        crit_opt.load_state_dict(data['critic_optim'])
        return data['epoch']
    except:
        return 0




def train(epochs, model_path, images_path, init_epoch=0, gen_freq=2):
    ntotal = len(REAL_LOADER)
    for epoch in range(init_epoch, init_epoch + epochs):
        since = time()
        for k, (images, labels) in enumerate(REAL_LOADER):
            z, label = random_batch(batch_size, input_size, labels)
            Gz = generator(z)
            label = torch.Tensor.repeat(label, (1, 1, 28, 28))
            DGz = critic(torch.cat([Gz, label], dim=1))
            Dx = critic(torch.cat([images, label], dim=1))
            
            crit_opt.zero_grad()
            crit_loss = torch.mean(DGz - Dx)
            crit_loss.backward()
            crit_opt.step()
            
            for _ in range(gen_freq):
                z, label = random_batch(batch_size, input_size, labels)
                Gz = generator(z)
                label = torch.Tensor.repeat(label, (1, 1, 28, 28))
                DGz = critic(torch.cat([Gz, label], dim=1))
                
                gen_opt.zero_grad()
                gen_loss = torch.mean(- DGz)
                gen_loss.backward()
                gen_opt.step()
        
           
            if k % 1000 == 999:   
                print(f'epoch: {epoch+1:4d}/{epochs+init_epoch}, iter: {k+1:6d}/{ntotal}, gen_loss: {gen_loss.item():7.3f}, critic_loss: {crit_loss.item():7.3f}, time: {time()-since:6.2f}')
            
        print(f'epoch: {epoch+1:4d}/{epochs+init_epoch}, iter: {k+1:6d}/{ntotal}, gen_loss: {gen_loss.item():7.3f}, critic_loss: {crit_loss.item():7.3f}, time: {time()-since:6.2f}')        
        if epoch % 5 == 4: 
            plt.figure()
            plt.suptitle(f'Epoch: {epoch+1:04d}')
            for i in range(1, 5):
                plt.subplot(2, 2, i)
                plt.imshow(Gz[i-1].detach().numpy().reshape(28, 28), cmap='gray', vmin=0, vmax=1)
                plt.title(f'{labels[i-1].item()}')
                plt.axis('off')
            plt.savefig(f'{images_path}/train_{epoch+1:04d}.png')
            data = {
                'generator': generator.state_dict(),
                'critic': critic.state_dict(),
                'gen_optim': gen_opt.state_dict(),
                'critic_optim': crit_opt.state_dict(),
                'epoch': epoch+1,
            }
            torch.save(data, f'{model_path}/CondGAN.pth')
            
            gen_sch.step()
            crit_sch.step()
            
                
if __name__ == '__main__':
    epoch = load_model('models')
    train(200, 'models', 'imgs', epoch)
            
            

