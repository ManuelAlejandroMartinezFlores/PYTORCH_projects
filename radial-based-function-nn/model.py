from turtle import forward
import torch 
import torch.nn as nn 
from torch.optim import Adam


class RBFLayer(nn.Module):
    def __init__(self, features=2, units=3):
        super(RBFLayer, self).__init__()
        self.c = nn.Parameter(torch.randn((units, features))).view(1, units, features)
        self.r = nn.Parameter(torch.randn(units))
        self.size = (units, features)
        
    def forward(self, x):
        x = x.unsqueeze(1).expand((x.size(0), ) + self.size)
        c = self.c.expand((x.size(0), ) + self.size)
        x = torch.exp(- torch.square(self.c - x).sum(-1) / self.r.square().unsqueeze(0))
        return x
    
class RBFNeuralNet(nn.Module):
    def __init__(self, in_features, units=4):
        super(RBFNeuralNet, self).__init__()
        self.rbflayer = RBFLayer(in_features, units)
        self.fc = nn.Linear(units, 1)
        
    def forward(self, x):
        x = self.rbflayer(x)
        return torch.sigmoid(self.fc(x))

        
if __name__ == '__main__':
    model = RBFNeuralNet(2, 4)
    
    optim = Adam(model.parameters(), lr=0.1)
    criteria = nn.BCELoss()
    
    x = torch.tensor(
        [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    ).type(torch.float)
    y = torch.tensor([[0, 1, 1, 0]]).view(-1, 1).type(torch.float)
    
    for k in range(200):
        out = model(x)
        loss = criteria(out, y)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
    print(model(x))
    
    x = torch.linspace(0, 1, 100).repeat(100, 1)
    y = torch.linspace(0, 1, 100).view(-1, 1).repeat(1, 100)
    z = torch.stack([x, y]).permute((1, 2, 0)).view(-1, 2)
    
    print(model.rbflayer.c)
    
    out = model(z)
    
    out = out.view(100, 100, 1).permute((0, 1, 2)).view(100, 100)
    
    import seaborn as sns 
    import matplotlib.pyplot as plt 
    
    plt.figure()
    sns.heatmap(out.detach().numpy(), xticklabels=[], yticklabels=[])
    plt.savefig('imgs/xor.png')
    
    
    