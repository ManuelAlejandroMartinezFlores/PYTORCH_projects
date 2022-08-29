from webbrowser import get
import torch 
from torch.utils.data import Dataset, DataLoader 
from PIL import Image 
import torchvision.transforms as T
import torch.nn.functional as F
import random

from create_data import ATEST, ATRAIN, BTRAIN, BTEST


class Tnorm:
    def __call__(self, x):
        return x / 255.


NORM = T.Compose([
    T.PILToTensor(),
    Tnorm(),
])

JITTER = T.Compose([
    T.RandomHorizontalFlip(),
    T.Resize(140),
    T.RandomCrop(128)
])

class ImageDataset(Dataset):
    def __init__(self, paths, train=True):
        self.imgs = paths
        self.n = len(paths)
        self.train = train
        
    def __getitem__(self, index):
        x = Image.open(self.imgs[index])
        x = NORM(x)
        x = JITTER(x)
        return x.unsqueeze(0), [1]
    
    def __len__(self):
        return self.n  
    
    
# ATRAIN = DataLoader(ImageDataset(ATRAIN, 0), batch_size=1, shuffle=True)
# ATEST = DataLoader(ImageDataset(ATEST, 0, False), batch_size=1, shuffle=True)
# BTRAIN = DataLoader(ImageDataset(BTRAIN, 1), batch_size=1, shuffle=True)
# BTEST = DataLoader(ImageDataset(BTEST, 1, False), batch_size=1, shuffle=True)   

ATRAIN = ImageDataset(ATRAIN)
ATEST = ImageDataset(ATEST, False)
BTRAIN = ImageDataset(BTRAIN)
BTEST = ImageDataset(BTEST, False)

def get_labels(classtype):
    return F.one_hot(torch.tensor([classtype]), num_classes=2).view(1, 2, 1, 1).repeat(1, 1, 256, 256)

ALABEL = get_labels(0)
BLABEL = get_labels(1)

if __name__ ==  '__main__':
    horses, _ = random.choice(ATRAIN)
    zebras, _ = random.choice(BTRAIN)
    print(horses.shape, zebras.shape)
    print(ALABEL.shape, BLABEL.shape)
    