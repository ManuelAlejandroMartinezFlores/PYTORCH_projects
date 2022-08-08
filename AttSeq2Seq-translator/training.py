
import torch 
import torch.nn as nn 
from torch.optim import Adam 
from torch.optim.lr_scheduler import StepLR

from time import time
import random


from torch_data import ids_to_sentence, iterate_dataloader
from model import Seq2SeqEncoder, AttSeq2SeqDecoder

LR = 2e-3

encoder = Seq2SeqEncoder()
decoder = AttSeq2SeqDecoder()

enc_optim = Adam(encoder.parameters(), lr=LR)
dec_optim = Adam(decoder.parameters(), lr=LR)

enc_sch = StepLR(enc_optim, step_size=100, gamma=0.97)
dec_sch = StepLR(dec_optim, step_size=100, gamma=0.97)

criteria = nn.NLLLoss()

TEACHER_RATIO = 0.9 
TEACHER_GAMMA = 0.97

def load_model():
    try:
        data = torch.load(f'models/AttSeq2Seq.pth')
        encoder.load_state_dict(data['encoder'])
        decoder.load_state_dict(data['decoder'])
        enc_optim.load_state_dict(data['enc_optim'])
        dec_optim.load_state_dict(data['dec_optim'])
        TEACHER_RATIO = data['teacher_ratio']
        return data['epoch'], TEACHER_RATIO
    except:
        return 0, 0.95
    
    
def train(epochs, init_epoch=0, TEACHER_RATIO=0.9):
    teacher_ratio = TEACHER_RATIO
    for epoch in range(init_epoch, epochs + init_epoch):
        k = 0
        since = time()
        for inputs, outputs in iterate_dataloader():
            loss = 0
            k += 1
            show_en = []
            show_sp = []
            show_pred = []
            for x, y in zip(inputs, outputs):
                enc_out, h, c = encoder(torch.tensor(x, dtype=torch.long))
                
                teacher_force = random.random() < teacher_ratio
                pred = []
                
                inputw = 0
                for w in y:
                    out, h, c = decoder(torch.tensor([inputw], dtype=torch.long), h, c, enc_out)
                    
                    inputw = torch.argmax(out, dim=1).detach().item()
                    loss += criteria(out, torch.tensor([w], dtype=torch.long))
                    
                    if k % 50 == 0: pred.append(inputw)
                    
                    if teacher_force:
                        inputw = w 
                        
                if k % 100 == 0:
                    show_sp.append(ids_to_sentence(x, 'sp'))
                    show_en.append(ids_to_sentence(y, 'en'))
                    show_pred.append(ids_to_sentence(pred, 'en'))
                    
                        
            enc_optim.zero_grad()
            dec_optim.zero_grad()
            loss.backward()
            dec_optim.step()
            enc_optim.step()
            enc_sch.step()
            dec_sch.step()
            
            if k % 100 == 0:
                print(f'\n \nepoch{epoch+1:4d}, iter: {k:4d}, loss: {loss.item():3.6f}, time: {time()-since:6.2f}, teacher_ratio: {teacher_ratio}')
                data = {
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'enc_optim': enc_optim.state_dict(),
                    'dec_optim': dec_optim.state_dict(),
                    'teacher_ratio': teacher_ratio,
                    'epoch': epoch+1
                }
                torch.save(data, 'models/AttSeq2Seq.pth')
                teacher_ratio = max(TEACHER_GAMMA * teacher_ratio, 0.4)
                
                for i, o, p in zip(show_sp[:4], show_en[:4], show_pred[:4]):
                    print('-----------------------------------')
                    print(f'input: {i}')
                    print(f'predicted: {p}')
                    print(f'output: {o}')
                
            
            
if __name__ == '__main__':
    epoch, TEACHER_RATIO = load_model()
    # train(200, epoch, TEACHER_RATIO)
    train(200, epoch)