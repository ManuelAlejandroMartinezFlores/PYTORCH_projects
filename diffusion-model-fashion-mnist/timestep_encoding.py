import torch 
import torch.nn as nn
import math



class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim=256):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.dim = dim 
        
    def forward(self, time):
        half_dim = self.dim // 2
        exponents = math.log(10000) / (half_dim - 1)
        denominators = torch.exp(- torch.arange(half_dim) * exponents)
        arguments = time[:, None] * denominators[None, :]
        return torch.cat([arguments.sin(), arguments.cos()], dim=-1)
    
    
    


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    time = torch.arange(100)
    time = time.type(torch.long)
    
    S = SinusoidalPositionEmbedding()
    y = S(time)
    print(y.shape)
    
    plt.figure()
    sns.heatmap(y)
    plt.axis('off')
    plt.title('Embedding')
    plt.savefig('imgs/embedding.png')
    