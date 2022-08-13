import torch 
import torch.nn as nn 
import torch.nn.functional as F
from time import time
import random


from torch_data import N, iterate_dataloader

MAX_LEN = 11

class Seq2SeqEncoder(nn.Module):
    def __init__(self, input_size=N['sp'], hidden_size=256, num_layers=1):
        super(Seq2SeqEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
    def forward(self, x):
        x = self.embed(x)
        out, (h, c) = self.lstm(x)
        return out, h, c 
    
    def evaluate(self, x):
        return self.forward(x)
        
    
    
class AttSeq2SeqDecoder(nn.Module):
    def __init__(self, input_size=N['en'], hidden_size=256, num_layers=1, output_size=N['en'], dropout_p=0.1):
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
        att_w = torch.softmax(att_w, dim=1)
        
        attn_app = torch.matmul(att_w, enc_out)
        attn_app = torch.cat([attn_app, embed], dim=1)
        attn_comb = F.relu(self.attn_comb(attn_app))
        
        out, (h, c) = self.lstm(attn_comb, (h, c))
        y = self.fc(out)
        y = torch.log_softmax(y, dim=1)
        return y, h, c 
    
    def evaluate(self, x, h, c, enc_out):
        x = self.embed(x)
        embed = self.dropout(x)
        
        attn = torch.cat([h[-1], c[-1], embed[0]], dim=0).view(1, -1)
        att_w = self.attn(attn)
        att_w = torch.softmax(att_w, dim=1)
        
        attn_app = torch.matmul(att_w[:, :enc_out.size(0)], enc_out)
        attn_app = torch.cat([attn_app, embed], dim=1)
        attn_comb = F.relu(self.attn_comb(attn_app))
        
        out, (h, c) = self.lstm(attn_comb, (h, c))
        y = self.fc(out)
        y = torch.log_softmax(y, dim=1)
        return y, h, c, att_w
        
    
    
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
            enc_out = torch.cat([enc_out, torch.zeros(MAX_LEN - enc_out.size(0), 256)])
            print(enc_out.shape)
            
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