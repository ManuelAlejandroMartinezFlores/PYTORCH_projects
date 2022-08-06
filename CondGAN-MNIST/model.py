import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

batch_size = 24
input_size = 100
# output 784 = 28 * 28
test_dataset = datasets.MNIST(root="./data", train=False,
                                           transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                           shuffle=False)


class CondGeneratorMNIST(nn.Module):
    def __init__(self, input_size=110):
        super(CondGeneratorMNIST, self).__init__()
        self.c1 = nn.ConvTranspose2d(input_size, 64, 6)                 # (1 - 1)1 + 6 = 6
        self.c2 = nn.ConvTranspose2d(64, 16, 3, stride=2)               # (6 - 1)2 + 3 = 13
        self.c3 = nn.ConvTranspose2d(16, 1, 4, stride=2)                # (13 - 1)2 + 4 = 28
        self.bn = nn.BatchNorm2d(16)

        
    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = self.bn(x)
        x = torch.sigmoid(self.c3(x))
        return x 
    
class CondCriticMNIST(nn.Module):
    def __init__(self): 
        super(CondCriticMNIST, self).__init__()                         # (I + 2P - K) / S + 1
        self.c1 = nn.Conv2d(11, 16, 4, stride=2)                          # (28 - 4) / 2 + 1 = 13
        self.c2 = nn.Conv2d(16, 32, 3, stride=2)                         # (13 - 3) / 2 + 1 = 6
        self.bn = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 64, 6)                                  
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = self.bn(x)
        x = F.relu(self.c3(x))
        x = self.flatten(x)
        return self.fc(x)


if __name__ == '__main__':
    gen = CondGeneratorMNIST()
    data = torch.randn(8, 100, 1, 1)
    label = torch.tensor([0, 2, 3, 4, 5, 1, 2, 8])
    label = F.one_hot(label, num_classes=10).view(-1, 10, 1, 1)
    print(data.shape, label.shape)
    data = torch.cat([data, label], dim=1)
    print(data.shape)
    y = gen(data)
    print(y.shape)
    label = torch.Tensor.repeat(label, (1, 1, 28, 28))
    print(label.shape)
    
    y = torch.cat([y, label], dim=1)
    print(y.shape)
    
    crit = CondCriticMNIST()
    z = crit(y)
    print(z.shape)
    