import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable



class RBM(nn.Module):
    def __init__(self, vis, hid, k=5):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(vis, hid) * 1e-2)
        self.v_bias = nn.Parameter(torch.zeros(vis))
        self.h_bias = nn.Parameter(torch.zeros(hid))
        self.k = k 
        
        
    def sample_from_p(self, p):
        return F.relu(torch.sign(p - Variable(torch.rand_like(p))))
    
    def h_to_v(self, h):
        h = F.linear(h, self.W, self.v_bias)
        ph = torch.sigmoid(h)
        return ph, self.sample_from_p(ph)
    
    def v_to_h(self, v):
        v = F.linear(v, self.W.t(), self.h_bias)
        vh = torch.sigmoid(v)
        return vh, self.sample_from_p(vh)
    
    def forward(self, x):
        pre_h1, h = self.v_to_h(x)
        
        for k in range(self.k):
            _, v = self.h_to_v(h)
            _, h = self.v_to_h(v)
            
        return x, v 
    
    def free_energy(self, x):
        vbias = x.mv(self.v_bias)
        wxb = F.linear(x, self.W.t(), self.h_bias)
        hidden = wxb.exp().add(1).log().sum(1)
        return - (vbias + hidden).mean()
    
    
    
if __name__ == '__main__':
    from torchvision import datasets 
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from time import time
    
    
    batch_size = 24
    input_size = 100
    # output 784 = 28 * 28
    test_dataset = datasets.MNIST(root="./data", train=False,
                                            transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                            shuffle=False)
    
    model = RBM(784, 512)
    flatten = nn.Flatten()
    
    since = time()
    imgs, _ = iter(test_loader).next()
    imgs = flatten(imgs)
    imgs = imgs.bernoulli()
    
    v, vk = model(imgs)
    
    loss = model.free_energy(v) - model.free_energy(vk)
    
    loss.backward()
    
    print(time() - since)