import torch 
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset 
import matplotlib.pyplot as plt
from torchvision.models import AlexNet_Weights
import random

from create_data import tokenize 

data = torch.load('data/dicts.pth')
WORDS2ID = data['w2i']
ID2WORDS = data['i2w']
N = len(ID2WORDS)


def sentence2id(x):
    return [WORDS2ID[w] for w in tokenize(x)]

class TextTransform:
    def __call__(self, x):
        return random.choice(x)
    
    


alex_trans = AlexNet_Weights.DEFAULT.transforms()

    
TRAIN = datasets.CocoCaptions('./data/imgs', './data/captions_val2017.json', transform=alex_trans, target_transform=TextTransform())

DATALOADER = DataLoader(TRAIN, batch_size=32, shuffle=True)


if __name__ == '__main__':
    
    for img, caption in DATALOADER:
        print(img.shape)
        for i, c in zip(img, caption):
            print(i.shape)
            print(c)
            break
        break