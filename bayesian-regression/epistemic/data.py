import torch 
from torch.utils.data import Dataset, DataLoader




class Data(Dataset):
    def __init__(self, n=600, yfunc=lambda x: x**3 - 2* x**2, nfunc=lambda x: x**2 - x):
        self.x = torch.linspace(-1, 2, n).view(n, -1)
        self.y = yfunc(self.x).view(n, -1)
        self.y += nfunc(self.x) * torch.randn_like(self.y)
        self.n = n 
        
    def __len__(self):
        return self.n 
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
def get_dataloader(**kwargs):
    data = Data(**kwargs)
    return DataLoader(data, batch_size=32, shuffle=True)
    
data = Data(yfunc=lambda x: 1 - 0.5 * x**2, nfunc=lambda x: 1/2)
DATA = get_dataloader(yfunc=lambda x: 1 - 0.5 * x**2, nfunc=lambda x: 1/2)


if __name__ == '__main__':
    import seaborn as sns 
    import matplotlib.pyplot as plt 
    
    plt.figure()
    sns.scatterplot(x=data.x.view(-1).detach().numpy(), y=data.y.view(-1).detach().numpy(), alpha=0.5)
    plt.title('Data')
    plt.savefig('imgs/data.png')
    
        