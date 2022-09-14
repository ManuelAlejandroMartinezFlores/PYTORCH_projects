import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from time import time 
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.optim import Adam
from torchvision import datasets 
from torch.utils.data import DataLoader
from torchvision import transforms as T

from model import DBN

trans = T.Compose([
    T.ToTensor(),
])
 
    
batch_size = 32
input_size = 100
# output 784 = 28 * 28
test_dataset = datasets.MNIST(root="./data", train=True,
                                        transform=trans)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                            shuffle=True)

def eval(imgs, out):
    y = torch.argmax(out, dim=1)
    imgs = imgs.view(-1, 28, 28).detach().numpy()
    plt.figure()
    for k in range(1, 13):
        plt.subplot(3, 4, k)
        plt.imshow(imgs[k-1], cmap='gray')
        plt.title(f'{y[k-1].item()}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('imgs/test.png')
        
        
    

def train(epochs, model, optim):
    flatten = nn.Flatten()
    # for layer in range(model.nlayers):
    #     model.set_train(layer, True)
    vk = 0
    imgs = 0
    
    for layer in range(model.nlayers):
        for epoch in range(epochs[layer]):
            since = time()
            loss_ = 0
            for imgs, labels in test_loader:
                imgs = flatten(imgs)
                imgsb = imgs.bernoulli()
                labels = F.one_hot(labels, num_classes=10)
                
                v, vk = model(imgsb, layer=layer, labels=None)
                
                loss = model.free_energy(v, layer) - model.free_energy(vk, layer)
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                loss_ += loss.item()
            
            loss_ /= len(test_loader)
            print(f'layer: {layer+1}, epoch: {epoch+1}, loss: {loss_:.5f}, time: {time()-since:.2f}')
            
        model.set_train(layer, False)
    
    imgs, _ = iter(test_loader).next()
    imgsb = flatten(imgs).bernoulli()
    _, vk = model(imgsb)        
    out = vk[:, -10:]
    eval(imgs, out)
    
        
            
def complete_train(epochs):
    model = DBN(784)
    optim = Adam(model.parameters(), lr=1e-2)
    for epoch in range(epochs):
        print(f'total epoch: {epoch+1:2d}')
        train([1] * 4, model, optim)

if __name__ == '__main__':
    complete_train(1)
    