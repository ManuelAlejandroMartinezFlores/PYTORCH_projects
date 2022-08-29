import pandas as pd 
import os 
import matplotlib.pyplot as plt
from PIL import Image


ATRAIN = ['data/trainA/' + p for p in os.listdir('data/trainA')]
ATEST = ['data/testA/' + p for p in os.listdir('data/testA')]
BTRAIN = ['data/trainB/' + p for p in os.listdir('data/trainB')]
BTEST = ['data/testB/' + p for p in os.listdir('data/testB')]



if __name__ == '__main__':
    plt.figure()
    k = 1
    for a, b in zip(ATRAIN, BTRAIN):
        plt.subplot(2, 2, k)
        a = Image.open(a)
        plt.imshow(a)
        plt.axis('off')
        
        plt.subplot(2, 2, k+1)
        b = Image.open(b)
        plt.imshow(b)
        plt.axis('off')
        
        k += 2
        if k == 5: break 
        
    plt.savefig('imgs/data.png')
        