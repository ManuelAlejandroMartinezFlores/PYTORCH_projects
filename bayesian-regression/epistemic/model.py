import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.distributions as D


class VariationalLinear(nn.Module):
    def __init__(self, infeatures=1, outfeatures=1):
        super(VariationalLinear, self).__init__()
        self.muW = nn.Parameter(torch.zeros(infeatures, outfeatures))
        self.sigmaW = nn.Parameter(torch.ones(infeatures, outfeatures))
        self.muB = nn.Parameter(torch.zeros(1, outfeatures))
        self.sigmaB = nn.Parameter(torch.ones(1, outfeatures))
        self.dW = D.Normal(self.muW, self.sigmaW)
        self.dB = D.Normal(self.muB, self.sigmaB)
        
    def forward(self, x):
        W = self.dW.sample()
        B = self.dB.sample()
        return F.linear(x, W.t(), B)

class Model(nn.Module):
    def __init__(self, units=8):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, units),
            nn.ReLU(),
            VariationalLinear(units, units),
            nn.ReLU(),
            nn.Linear(units, 1),
        )
        
        
    def forward(self, x):
        return self.model(x)
    
    
if __name__ == '__main__':
    from data import DATA 
    
    model = Model()
    criteria = nn.MSELoss()
    
    x, y = iter(DATA).next()
    
    out = model(x)
    
    loss = criteria(out, y)
    
    loss.backward()