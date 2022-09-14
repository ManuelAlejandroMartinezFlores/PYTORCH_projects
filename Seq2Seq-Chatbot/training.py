import time
import torch 
import torch.nn as nn 
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import random

from torch_data import GLOVE, ID_TO_WORDS, WORDS_TO_ID, iterate_dataloader, NITERS
from model import Seq2SeqDecoder, Seq2SeqEncoder 



LR = 1e-4
TEACHER_RATIO = 0.9

encoder = Seq2SeqEncoder()
decoder = Seq2SeqDecoder() 

enc_optim = Adam(encoder.parameters(), lr=LR)
dec_optim = Adam(decoder.parameters(), lr=LR)
enc_sc = StepLR(enc_optim, step_size=1000, gamma=0.97)
dec_sc = StepLR(dec_optim, step_size=1000, gamma=0.97)
criteria = nn.NLLLoss()

def load_model():
    try:
        DATA = torch.load('models/Seq2Seq.pth')
        encoder.load_state_dict(DATA['encoder'])
        decoder.load_state_dict(DATA['decoder'])
        enc_optim.load_state_dict(DATA['enc_optim'])
        enc_optim.load_state_dict(DATA['dec_optim'])
        TEACHER_RATIO = DATA['teacher_ratio']
        return DATA['epoch'], TEACHER_RATIO
    except:
        return 0


def train(epochs, init_epoch=0, TEACHER_RATIO=0.95):
    teacher_ratio = TEACHER_RATIO
    for epoch in range(init_epoch, init_epoch + epochs):
        since = time.time()
        show_ins = []
        show_outs = []
        show_preds = []
        teacher_gamma = 0.97
        k = 0
        for inputs, outputs in iterate_dataloader():
            loss = 0
            k += 1
            for x, y in zip(inputs, outputs):
                enc_h, enc_c = encoder.initHidden()
                out, enc_h, enc_c = encoder(x, enc_h, enc_c)
                    
                dec_h, dec_c = enc_h.view(3, -1), enc_c.view(3, -1)
                pred = []
                inputw = '<BOS>'
                
                teacher_force = random.random() < teacher_ratio
                for w in y:
                    if w not in WORDS_TO_ID:
                        continue
                    out, dec_h, dec_c = decoder(inputw, dec_h, dec_c)
                    
                    prediction = torch.argmax(out.detach(), dim=0)
                    inputw = ID_TO_WORDS[prediction.detach().item()]
                    
                    if k % 200 == 0:
                        pred.append(inputw)
                        
                    if teacher_force:
                        inputw = w 
                    
                    loss += criteria(out.view(1, -1), torch.tensor([WORDS_TO_ID[w]], dtype=torch.long)) / len(y)
                        
                if k % 200 == 0:
                    show_ins.append(' '.join(x))
                    show_outs.append(' '.join(y))   
                    show_preds.append(' '.join(pred)) 
                    
            dec_optim.zero_grad()
            enc_optim.zero_grad()
            loss = loss / 32
            loss.backward()
            dec_optim.step()
            enc_optim.step()
            enc_sc.step()
            dec_sc.step()
            
            
                    
            if k % 200 == 0:
                print(f'\n \nepoch: {epoch+1:4d}, iter {k:6d}, loss: {loss.item():.6f}, time: {time.time() - since}, teacher ratio: {teacher_ratio:.6f}')
            
            
                for i, o, p in zip(show_ins[:4], show_outs[:4], show_preds[:4]):
                    print('------------------------')
                    print(f'input: {i}')
                    print(f'predicted: {p}')
                    print(f'output: {o}')
                    
                show_ins = []
                show_outs = []
                show_preds = []
            
                DATA = {
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'enc_optim': enc_optim.state_dict(),
                    'dec_optim': dec_optim.state_dict(),
                    'teacher_ratio': teacher_ratio,
                    'epoch': epoch+1,
                }
                torch.save(DATA, 'models/Seq2Seq.pth')
                teacher_ratio = max(teacher_gamma * teacher_ratio, 0.4)

            
        
        print(f'\n \nepoch: {epoch+1}, loss: {loss.item():.6f}, time: {time.time() - since}')
        
        DATA = {
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'enc_optim': enc_optim.state_dict(),
            'dec_optim': dec_optim.state_dict(),
            'teacher_ratio': teacher_ratio,
            'epoch': epoch+1,
        }
        torch.save(DATA, 'models/Seq2Seq.pth')
        
        
if __name__ == '__main__':
    epoch, TEACHER_RATIO = load_model()
    train(200, epoch, TEACHER_RATIO)
        




