import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage 
import random

from model_unet import UNet
from torch_data import ATEST, BTEST

t = ToPILImage()

def load():
    to_horse = UNet()
    to_zebra = UNet()
    try:
        data = torch.load('models/cycle-gan.pth') # CycleGAN-horse2zebra/models/cycle-gan.pth
        to_horse.load_state_dict(data['to_horse'])
        to_zebra.load_state_dict(data['to_zebra'])
    except:
        print('error')
    return to_horse, to_zebra



def evaluate(filename):
    to_horse, to_zebra = load()
    
    horses, _ = random.choice(ATEST)
    zebras, _ = random.choice(BTEST)
    
    with torch.no_grad():
    
        gen_horse = to_horse(zebras)
        gen_zebra = to_zebra(horses)
        
        rev_horse = to_horse(gen_zebra)
        rev_zebra = to_zebra(gen_horse)
        
        plt.figure()
        
        plt.subplot(2, 3, 1)
        plt.imshow(t(horses[0]))
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(t(gen_zebra[0]))
        plt.title('Forward')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(t(rev_horse[0]))
        plt.title('Reverse')
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.imshow(t(zebras[0]))
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(t(gen_horse[0]))
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        plt.imshow(t(rev_zebra[0]))
        plt.axis('off')
        
        plt.savefig(filename)
    
if __name__ == '__main__':
    for k in range(1, 6):
        evaluate(f'imgs/ev{k:02d}.png')
        
        
    
    