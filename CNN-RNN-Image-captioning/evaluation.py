from torchvision.models import alexnet, AlexNet_Weights
import torch.nn as nn
import torch
from torch.optim import Adam 
from torch.optim.lr_scheduler import StepLR
import random
import matplotlib.pyplot as plt 
from torchvision.models import AlexNet_Weights
from textwrap import wrap


from model import Encoder, Decoder
from torch_data import sentence2id, ID2WORDS
from create_data import TRAIN 

alex_trans = AlexNet_Weights.DEFAULT.transforms()


def load():
    encoder = Encoder()
    decoder = Decoder()
    try:
        data = torch.load('models/CNN-RNN-ImgCapt.pth')
        encoder.load_state_dict(data['encoder'])
        decoder.load_state_dict(data['decoder'])
    except:
        pass
    return encoder, decoder



def evaluate(filename):
    encoder, decoder = load()
    
    k = 1 
    
    plt.figure()
    while k < 5:
        plt.subplot(2, 2, k)
        image, caption = random.choice(TRAIN)
        caption = random.choice(caption)
        plt.imshow(image)
        plt.axis('off')
        
        img = alex_trans(image).view(1, 3, 224, 224)
        img_out = encoder(img)
        h, c = decoder.init_hidden()
        
        pred = []
        inputw = 0
        for w in sentence2id(caption) + [1]:
            out, h, c = decoder(torch.tensor([inputw]), h, c, img_out)
            inputw = torch.argmax(out, dim=1).item()
            pred.append(ID2WORDS[inputw])
            
        title = '\n'.join(wrap(' '.join(pred), 40))
        plt.title(title, fontsize=8)
        
        k += 1
        
    plt.tight_layout(pad=3)
    plt.savefig(filename)
    plt.close()
    
    
if __name__ == '__main__':
    for k in range(1, 6):
        evaluate(f'imgs/ev{k:02d}.png')
        
    
