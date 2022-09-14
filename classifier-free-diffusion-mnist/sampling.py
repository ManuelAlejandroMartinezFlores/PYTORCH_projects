import torch 
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms as T
# import seaborn as sns


from model_unet import UNet 
from noise_scheduler import get_betas 
from data import train_dataset

trans = T.Compose([
    T.Lambda(lambda x: (x + 1) / 2),
    T.ToPILImage()
])

class Sampler:
    def __init__(self, model:UNet, steps=100, size=(1, 28, 28), min=1e-6, max=1 - 1e-6, classes=train_dataset.classes,
                 w=3.5):
        self.model = model
        self.betas = get_betas(steps, min, max) 
        self.alphas = 1. - self.betas 
        self.cum_alphas = self.alphas.cumprod(axis=0)
        cum_alphas_prev = F.pad(self.cum_alphas[:-1], (1, 0), value=1)
        self.post_var = self.betas * (1. - cum_alphas_prev) / (1. - self.cum_alphas)
        self.size = size
        self.steps = steps
        self.classes = classes
        self.w = w
        
    def sample_step(self, x, t, y):
        with torch.no_grad():
            ehat = (self.w + 1) * self.model(x, t, y, torch.ones_like(y)) - self.w * self.model(x, t)
            mean = x - ehat * self.betas[t].view(x.size(0), 1, 1, 1) / torch.sqrt(1. - self.cum_alphas[t]).view(x.size(0), 1, 1, 1)
            mean /=  torch.sqrt(self.alphas[t]).view(x.size(0), 1, 1, 1)
            if t[0].item() == 0:
                return mean
            return mean + self.post_var[t].view(x.size(0), 1, 1, 1) * torch.randn_like(x)
        
    def sample(self, y, steps=8, samples=4):
        size = (samples, ) + self.size
        x = torch.randn(size)
        
        imgs = []
        steps_ = steps
        if self.steps % steps != 0: 
            steps_ = steps - 1
        every = self.steps // steps_
        for t in range(0, self.steps)[::-1]:
            x = self.sample_step(x, torch.tensor([t] * samples), y)
            if t % every == 0:
                imgs.append(x)
        

        assert len(imgs) == steps
                
        return imgs 
    
    
    def plot_samples(self, filename, steps=7, samples=4):
        y = torch.randint(1, len(self.classes), (samples, ))
        imgs = self.sample(y, steps)
        
        plt.figure()
        for r in range(samples):
            for c in range(steps):
                plt.subplot(samples, steps, r * steps + c + 1)
                plt.imshow(trans(imgs[c][r]), cmap='gray')
                if c == 3:
                    plt.title(self.classes[y[r].item()])
                plt.axis('off')
                
        plt.savefig(filename)
                
def load(model:UNet):
    try:
        data = torch.load('models/diffusion-mnist.pth')
        model.load_state_dict(data['model'])
    except:
        pass 

if __name__ == '__main__':
    model = UNet()
    load(model)
    S = Sampler(model, max=0.02)
    for k in range(1, 6):
        S.plot_samples(f'imgs/ev{k:02d}.png')