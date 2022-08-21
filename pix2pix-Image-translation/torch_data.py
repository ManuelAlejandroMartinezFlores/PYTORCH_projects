
from torchvision import transforms as T 
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from create_data import TRAIN, TEST 

class Norm:
    def __call__(self, x):
        return (x / 255)


p2t = T.PILToTensor()

transform = T.Compose([
    T.Resize(286),
    T.RandomCrop(256),
    T.RandomHorizontalFlip(),
    Norm()
])

class FacadeDataset(Dataset):
    def __init__(self, df:pd.DataFrame):
        self.n = len(df)
        self.df = df
        
    def __getitem__(self, index):
        df = self.df.loc[index]
        x = p2t(df['segmentation'])
        y = p2t(df['photo'])
        total = torch.cat([x, y], dim=0)
        total = transform(total)
        x = total[:3]
        y = total[3:]
        return x, y
    
    def __len__(self):
        return self.n

TRAINLOADER = DataLoader(FacadeDataset(TRAIN), batch_size=2, shuffle=True)
TESTLOADER = DataLoader(FacadeDataset(TEST), batch_size=32, shuffle=True)

if __name__ == '__main__':
    
    x, y = iter(TRAINLOADER).next()
    print(x.shape, y.shape)
    
    plt.figure()
    k = 1
    
    for seg, ph in zip(x, y):
        print(seg.shape)
        plt.subplot(2, 2, k)
        seg = T.ToPILImage()(seg)
        plt.imshow(seg)
        plt.axis('off')
        
        plt.subplot(2, 2, k+1)
        ph = T.ToPILImage()(ph)
        plt.imshow(ph)
        plt.axis('off')
    
        k += 2
        if k == 5: break
        
    plt.tight_layout()
    plt.savefig('imgs/transforms.png')