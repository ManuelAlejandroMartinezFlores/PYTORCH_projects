from torchvision.models import alexnet, AlexNet_Weights
import torch.nn as nn
import torch
from torch.optim import Adam 
from torch.optim.lr_scheduler import StepLR
import random
from time import time
import numpy as np


from model import Encoder, Decoder
from torch_data import DATALOADER, sentence2id, N
from evaluation import evaluate



def load(encoder:Encoder, decoder:Decoder, enc_optim:Adam, dec_optim:Adam):
    try:
        data = torch.load('models/CNN-RNN-ImgCapt.pth')
        encoder.load_state_dict(data['encoder'])
        decoder.load_state_dict(data['decoder'])
        enc_optim.load_state_dict(data['enc_optim'])
        dec_optim.load_state_dict(data['dec_optim'])
        return data['epoch'], data['teacher_ratio']
    except:
        return 0, 0.9

def save(encoder:Encoder, decoder:Decoder, enc_optim:Adam, dec_optim:Adam, epoch, teacher_ratio):
    data = {
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'enc_optim': enc_optim.state_dict(),
        'dec_optim': dec_optim.state_dict(),
        'epoch': epoch,
        'teacher_ratio': teacher_ratio
    } 
    torch.save(data, 'models/CNN-RNN-ImgCapt.pth')


def train(epochs, lr=1e-4, teacher_ratio=0.9, teacher_lambda=0.98):
    encoder = Encoder()
    decoder = Decoder()
    enc_optim = Adam(encoder.parameters(), lr=lr)
    dec_optim = Adam(decoder.parameters(), lr=lr)
    init_epoch, teacher_ratio = load(encoder, decoder, enc_optim, dec_optim)
    
    enc_sch = StepLR(enc_optim, step_size=1, gamma=0.97)
    dec_sch = StepLR(dec_optim, step_size=1, gamma=0.97)
    
    criteria = nn.CrossEntropyLoss()
    
    for epoch in range(init_epoch, init_epoch + epochs):
        epoch_loss = 0
        since = time()
        k = 0
        for inputs, outputs in DATALOADER:
            loss = 0 
            for img, caption in zip(inputs, outputs):
                img_out = encoder(img.view(1, 3, 224, 224))
                h, c = decoder.init_hidden()
                
                teacher_force = random.random() < teacher_ratio
                inputw = 0 
                
                if teacher_force:
                    x = torch.tensor([0] + sentence2id(caption))
                    y = torch.tensor(sentence2id(caption) + [1])
                    
                    out, _, _ = decoder(x, h, c, torch.Tensor.repeat(img_out, (y.size(0), 1)))
                    loss += criteria(out, y)
                else:
                    for w in sentence2id(caption) + [1]:
                        out, h, c = decoder(torch.tensor([inputw]), h, c, img_out)
                        
                        # inputw = torch.argmax(out, dim=1)
                        loss += criteria(out, torch.tensor([w])) / len(caption)

                        outnp = torch.softmax(out, dim=1).detach().numpy().reshape(N)
                        inputw = np.random.choice(N, p=outnp)
                        
            
            enc_optim.zero_grad()
            dec_optim.zero_grad()
            loss = loss / 32
            epoch_loss += loss.item()
            loss.backward()
            enc_optim.step()
            dec_optim.step()
            
            
            k += 1 

                
            
        
            
        teacher_ratio = max(teacher_ratio * teacher_lambda, 0.9)   
        
        epoch_loss /= len(DATALOADER)
        print(f'epoch: {epoch+1}, loss: {epoch_loss:3.4f}, time: {time()-since:5.2f}, teacher_ratio: {teacher_ratio:4f}')
        save(encoder, decoder, enc_optim, dec_optim, epoch+1, teacher_ratio)
        
        if epoch < 70:
            enc_sch.step()
            dec_sch.step()
        
        if epoch % 10 == 9:
            evaluate(f'imgs/epoch{epoch+1:04d}.png')
        
        
        
        
if __name__ == '__main__':
    train(500)
            
            
        