import torch 
import torch.nn as nn 
import torch.nn.functional as F
from time import time
import random


from torch_data import N, iterate_dataloader

MAX_LEN = 16

class Seq2SeqEncoder(nn.Module):
    def __init__(self, input_size=N['sp'], hidden_size=256, num_layers=3):
        super(Seq2SeqEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
    def forward(self, x):
        x = self.embed(x)
        out, (h, c) = self.lstm(x)
        return out, h, c 
        
    
    
class AttSeq2SeqDecoder(nn.Module):
    def __init__(self, input_size=N['en'], hidden_size=256, num_layers=3, output_size=N['en'], dropout_p=0.4):
        super(AttSeq2SeqDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers 
        
        self.embed = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        
        self.attn = nn.Linear(hidden_size * 3, MAX_LEN)
        self.attn_comb = nn.Linear(hidden_size * 2, hidden_size)
        
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h, c, enc_out):
        x = self.embed(x)
        embed = self.dropout(x)
        
        attn = torch.cat([h[-1], c[-1], embed[0]], dim=0).view(1, -1)
        att_w = self.attn(attn)
        
        attn_app = torch.matmul(att_w[:, :enc_out.size(0)], enc_out)
        attn_app = torch.cat([attn_app, embed], dim=1)
        attn_comb = F.relu(self.attn_comb(attn_app))
        
        out, (h, c) = self.lstm(attn_comb, (h, c))
        y = self.fc(out)
        y = torch.log_softmax(y, dim=1)
        return y, h, c 
        
    
    
if __name__ == '__main__':
    enc = Seq2SeqEncoder()
    dec = AttSeq2SeqDecoder()
    criteria = nn.NLLLoss()
    teacher_ratio = 0.5
    for inputs, outputs in iterate_dataloader():
        loss = 0
        since = time()
        for x, y in zip(inputs, outputs):
            enc_out, h, c = enc(torch.tensor(x, dtype=torch.long))
            print(len(x), enc_out.shape, h.shape, c.shape)
            
            teacher_force = random.random() < teacher_ratio
            
            inputw = 0
            for w in y:
                out, h, c = dec(torch.tensor([inputw], dtype=torch.long), h, c, enc_out)
                print(out.shape, h.shape, c.shape)
                
                inputw = torch.argmax(out, dim=1).detach().item()
                loss += criteria(out, torch.tensor([w], dtype=torch.long))
                
                if teacher_force:
                    inputw = w 
            
        loss.backward()   
        print(loss.item())
        print(time() - since)
        break