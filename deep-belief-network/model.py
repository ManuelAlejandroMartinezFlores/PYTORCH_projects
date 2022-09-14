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
    
    
    
class DBN(nn.Module):
    def __init__(self, vis=256, hid=[500, 500, 2000], ik=5, nclasses=10):
        super(DBN, self).__init__()
        self.rbms = []
        self.nlayers = len(hid)
        self.nclasses = nclasses
        
        for k in range(len(hid)):
            if k == 0:
                inp = vis 
            else:
                inp = hid[k-1]
            self.rbms.append(RBM(inp, hid[k], ik))
            
            
        
    def forward(self, v, layer=0, labels=None):
        if labels is None:
            labels = torch.zeros(v.size(0), self.nclasses)
        if layer <= 0: layer = self.nlayers - 1
        for k in range(layer):
            pv, v = self.rbms[k].v_to_h(v)
        if layer == self.nlayers - 1:
            v = torch.cat([v, labels], dim=1)
        return self.rbms[layer](v)
        
    def free_energy(self, v, layer=0):
        if layer <= 0: layer = self.nlayers - 1
        return self.rbms[layer].free_energy(v)
    
    def set_train(self, layer, train=False):
        self.rbms[layer].requires_grad_(train)
        
        

if __name__ == '__main__':
    from torchvision import datasets 
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from time import time
    
    
    batch_size = 32
    input_size = 100
    # output 784 = 28 * 28
    test_dataset = datasets.MNIST(root="./data", train=False,
                                            transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                            shuffle=False)
    
    model = DBN(784)
    flatten = nn.Flatten()
    
    for layer in range(model.nlayers):
        print(layer)
        since = time()
        imgs, labels = iter(test_loader).next()
        labels = F.one_hot(labels, num_classes=10)
        imgs = flatten(imgs)
        imgs = imgs.bernoulli()
        
        v, vk = model(imgs, layer, labels=labels)
        loss = model.free_energy(v, layer) - model.free_energy(vk, layer)
        
        loss.backward()
        
        
        print(time() - since)
                
        