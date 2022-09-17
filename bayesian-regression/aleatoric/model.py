import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.distributions as D


class Model(nn.Module):
    def __init__(self, input=1, hidden=64):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.sigma = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Softplus()
        )

        
    
    def forward(self, x):
        out = self.model(x)
        mu = self.mu(out)
        sigma = self.sigma(out)
        return D.Normal(mu, sigma)
    
    
    
class NLLLoss(nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()
        
    def forward(self, dist:D.Distribution, y):
        return - dist.log_prob(y).mean()

if __name__ == '__main__':
    from data import DATA 
    
    model = Model()
    criteria = NLLLoss()
    
    x, y = iter(DATA).next()
    
    dist = model(x)
    
    loss = criteria(dist, y)
    
    loss.backward()
    
