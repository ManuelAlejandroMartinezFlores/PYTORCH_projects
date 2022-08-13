from torchvision.models import alexnet, AlexNet_Weights
import torch.nn as nn
import torch
from torch.optim import Adam 
from torch.optim.lr_scheduler import StepLR
import random


from model import Encoder, Decoder
from torch_data import DATALOADER, sentence2id


def load(encoder:Encoder, decoder:Decoder, enc_optim:Adam, dec_optim:Adam):
    try:
        data = torch.load('models/CNN-RNN-ImgCapt.pth')
        encoder.load_state_dict(data['encoder'])
        decoder.load_state_dict(data['decoder'])
        enc_optim.load_state_dict(data['enc_optim'])
        dec_optim.load_state_dict(data['dec_optim'])
        return data['epoch']
    except:
        return 0

def save(encoder:Encoder, decoder:Decoder, enc_optim:Adam, dec_optim:Adam, epoch):
    data = {
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'enc_optim': enc_optim.state_dict(),
        'dec_optim': dec_optim.state_dict(),
        'epoch': epoch
    } 
    torch.save(data, 'models/CNN-RNN-ImgCapt.pth')


def train(epochs, lr=1e-3, teacher_ratio=0.5):
    encoder = Encoder()
    decoder = Decoder()
    enc_optim = Adam(encoder.parameters(), lr=lr)
    dec_optim = Adam(decoder.parameters(), lr=lr)
    init_epoch = load(encoder, decoder, enc_optim, dec_optim)
    
    enc_sch = StepLR(enc_optim, step_size=300, gamma=0.97)
    dec_sch = StepLR(dec_optim, step_size=300, gamma=0.97)
    
    criteria = nn.NLLLoss()
    
    for epoch in range(init_epoch, init_epoch + epochs):
        epoch_loss = 0
        for inputs, outputs in DATALOADER:
            loss = 0 
            for img, caption in zip(inputs, outputs):
                out = encoder(img).view(6, -1)
                h, c = out[:3], out[3:]
                
                teacher_force = random.random() < teacher_ratio
                inputw = 0 
                for w in sentence2id(caption) + [1]:
                    out = decoder(torch.tensor([inputw]), h, c)
                    inputw = torch.argmax(out, dim=1)
                    loss += criteria(out, torch.tensor([w])) / len(caption)
                    
                    if teacher_force:
                        inputw = w
                        
            
            enc_optim.zero_grad()
            dec_optim.zero_grad()
            loss = loss / 32
            epoch_loss += loss.item()
            loss.backward()
            enc_optim.step()
            dec_optim.step()
            enc_sch.step()
            dec_sch.step()
            
            
        save(encoder, decoder, enc_optim, dec_optim, epoch+1)
        loss_epoch /= len(DATALOADER)
        print(f'epoch: {epoch+1}, loss: {epoch_loss:3.4f}')
            
            
        