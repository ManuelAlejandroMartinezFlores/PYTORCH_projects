import torch 
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from time import time
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


from model import RBM


def train(epochs):
    batch_size = 32
    # output 784 = 28 * 28
    train = datasets.MNIST(root="./data", train=True,
                                            transform=transforms.ToTensor())
    DATA = DataLoader(dataset=train, batch_size=batch_size,
                                            shuffle=False)
    
    model = RBM(784, 512)
    flatten = nn.Flatten()
    optim = Adam(model.parameters(), lr=1e-2)
    
    
    for epoch in range(epochs):
        since = time()
        loss_ = 0
        for imgs, _ in DATA:
            imgs = flatten(imgs)
            imgs = imgs.bernoulli()
            
            v, vk = model(imgs)
            
            loss = model.free_energy(v) - model.free_energy(vk)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            loss_ += loss.item()
            
        loss_ /= len(DATA)
        
        print(f'epoch: {epoch+1:4d}, loss: {loss_:.3f}, time: {time() - since:.2f}')
        
        
    real = make_grid(v.view(32, 1, 28, 28).data)
    plt.figure()
    plt.imshow(real.detach().numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.title('Real')
    plt.savefig('imgs/real.png')
    
    fake = make_grid(vk.view(32, 1, 28, 28).data)
    plt.figure()
    plt.imshow(fake.detach().numpy().transpose((1, 2, 0)))
    plt.title('Fake')
    plt.axis('off')
    plt.savefig('imgs/fake.png')
    
        
if __name__ == '__main__':
    train(3)