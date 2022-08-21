
import pandas as pd 
import os
import matplotlib.pyplot as plt
from PIL import Image
import random


def get_photo_path(n, split='train'):
    return os.path.join('data', split + 'A',f'{n}{"_A" if split == "train" else ""}.jpg')

def get_seg_path(n, split='train'):
    return os.path.join('data', split + 'B', f'{n}{"_B" if split == "train" else ""}.jpg')


TRAIN = pd.DataFrame({'photo_path': [get_photo_path(n) for n in range(1, 401)], 
                      'segmentation_path': [get_seg_path(n) for n in range(1, 401)]})
TEST = pd.DataFrame({'photo_path': [get_photo_path(n, 'test') for n in range(1, 107)], 
                      'segmentation_path': [get_seg_path(n, 'test') for n in range(1, 107)]})

TRAIN['photo'] = TRAIN['photo_path'].map(lambda x: Image.open(x))
TRAIN['segmentation'] = TRAIN['segmentation_path'].map(lambda x: Image.open(x))

TEST['photo'] = TEST['photo_path'].map(lambda x: Image.open(x))
TEST['segmentation'] = TEST['segmentation_path'].map(lambda x: Image.open(x))



def load(index):
    df = TRAIN.loc[index, :]
    x = [Image.open(p) for p in df['photo']]
    y = [Image.open(p) for p in df['segmentation']]
    return x, y

if __name__ == '__main__':
    plt.figure()
    k = 1
    while k < 5:
        n = random.choice(range(400))
        plt.subplot(2, 2, k)
        img = Image.open(get_photo_path(n))
        print(img.size)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Facade')
        
        plt.subplot(2, 2, k+1)
        img = Image.open(get_seg_path(n))
        print(img.size)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Segmentation')
        
        k += 2
        
    plt.tight_layout()
    plt.savefig('imgs/data.png')
    
    print(TRAIN.head())
    
    # index = [1, 4, 6]
    # x, y = load(index)
    
    
    # for p in TRAIN['photo']:
    #     Image.open(p)
        
    # for p in TRAIN['segmentation']:
    #     Image.open(p)
        
    # for p in TEST['photo']:
    #     Image.open(p)
        
    # for p in TEST['segmentation']:
    #     Image.open(p)
        
