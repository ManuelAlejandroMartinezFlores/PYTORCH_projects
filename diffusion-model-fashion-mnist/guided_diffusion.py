import torch 
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms as T
# import seaborn as sns


from model_unet import UNet 
from noise_scheduler import get_betas 
from classifier import Discriminator 

from data import train_dataset, REAL_LOADER

trans = T.Compose([
    T.Lambda(lambda x: (x + 1) / 2),
    T.ToPILImage()
])

class GuidedSampler:
    def __init__(self, model:UNet, classifier:Discriminator, steps=100, size=(1, 28, 28), min=1e-6, max=1 - 1e-6,
                 classes = train_dataset.classes, s=10):
        self.betas = get_betas(steps, min, max)
        self.alphas = 1. - self.betas 
        self.cum_alphas = self.alphas.cumprod(dim=0)
        cum_alphas_prev = F.pad(self.cum_alphas[:-1], (1, 0), value=1)
        self.post_var = self.betas * (1. - cum_alphas_prev) / (1. - self.cum_alphas)
        self.model = model 
        self.classifier = classifier
        self.size = size
        self.classes = classes
        self.steps = steps
        self.s = s
        
        
    def sample_step(self, x, t, y):
        x = x.detach().requires_grad_(True)
        class_out = torch.log(torch.softmax(self.classifier(x, t), dim=1)[:, y]) 
        class_out.backward()
        x_grad = x.grad.clone().detach() * self.s
        
        with torch.no_grad():
            ehat = self.model(x, t) - torch.sqrt(1. - self.cum_alphas[t]).view(x.size(0), 1, 1, 1) * x_grad 
            ans = torch.sqrt(self.cum_alphas[t-1]).view(x.size(0), 1, 1, 1) * (x -
                            torch.sqrt(1. - self.cum_alphas[t]).view(x.size(0), 1, 1, 1) * ehat) / torch.sqrt(self.cum_alphas[t]).view(x.size(0), 1, 1, 1)
            return ans + torch.sqrt(1. - self.cum_alphas[t-1]).view(x.size(0), 1, 1, 1) * ehat
    
    def sample_step2(self, x, t, y):
        x = x.detach().requires_grad_(True)
        class_out = torch.log(torch.softmax(self.classifier(x, t), dim=1)[:, y]) 
        class_out.backward()
        x_grad = x.grad.clone().detach() * self.s
        
        with torch.no_grad():
            ehat = self.model(x, t) - torch.sqrt(1. - self.cum_alphas[t]).view(x.size(0), 1, 1, 1) * x_grad 
            mean = x - ehat * self.betas[t].view(x.size(0), 1, 1, 1) / torch.sqrt(1. - self.cum_alphas[t]).view(x.size(0), 1, 1, 1)
            mean /=  torch.sqrt(self.alphas[t]).view(x.size(0), 1, 1, 1)
            if t[0].item() == 0:
                return mean
            return mean + self.post_var[t].view(x.size(0), 1, 1, 1) * torch.randn_like(x)
    
    def sample(self, y, steps=8, samples=1):
        size = (samples, ) + self.size
        x = torch.randn(size)
        
        imgs = []
        steps_ = steps
        if self.steps % steps != 0: 
            steps_ = steps - 1
        every = self.steps // steps_
        for t in range(0, self.steps)[::-1]:
            x = self.sample_step2(x, torch.tensor([t] * samples), y)
            if t % every == 0:
                imgs.append(x)
                
        #imgs.append(x)
        

        assert len(imgs) == steps
                
        return imgs 
    
    def plot_samples(self, filename, steps=7, samples=4):
        
        plt.figure()
        for r in range(samples):
            y = torch.randint(0, len(self.classes), (1, )).type(torch.long)
            imgs = self.sample(y)
            for c in range(steps):
                plt.subplot(samples, steps, r * steps + c + 1)
                plt.imshow(trans(imgs[c][0]), cmap='gray')
                if c == 3:
                    plt.title(self.classes[y.item()])
                plt.axis('off')
                
        plt.savefig(filename)
    
    
def load(model:UNet, classifier:Discriminator):
    try:
        data = torch.load('models/diffusion-mnist.pth')
        model.load_state_dict(data['model'])
        data = torch.load('models/classifier.pth')
        classifier.load_state_dict(data['model'])
        print('models loaded')
    except:
        pass
    
    
if __name__ == '__main__':
    model, classifier = UNet(), Discriminator()
    load(model, classifier)
    GS = GuidedSampler(model, classifier, max=0.02)
    
    for k in range(1, 6):
        GS.plot_samples(f'imgs/guided_ev{k:02d}.png')
    
    


