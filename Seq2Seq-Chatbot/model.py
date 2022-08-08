import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch_data import GLOVE, ID_TO_WORDS, WORDS_TO_ID





class Seq2SeqEncoder(nn.Module):
    def __init__(self, input_size=100, hidden_size=256, num_layers=3):
        super(Seq2SeqEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def forward(self, x, h, c):
        x = GLOVE.get_vecs_by_tokens(x)
        out, (h, c) = self.lstm(x)
        return out, h, c 
        
        
    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size), torch.zeros(self.num_layers, 1, self.hidden_size)
    
    
class Seq2SeqDecoder(nn.Module):
    def __init__(self, input_size=100, output_size=len(WORDS_TO_ID), hidden_size=256, num_layers=3):
        super(Seq2SeqDecoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        
    def forward(self, x, h, c):
        x = GLOVE.get_vecs_by_tokens(x).view(1, -1)
        x = F.relu(x)
        out, (h, c) = self.lstm(x, (h, c))
        y = self.fc(out[-1,:])
        y = F.log_softmax(y, dim=0)
        return y, h, c 
        
        







if __name__ == '__main__':
    from torch_data import iterate_dataloader 
    from time import time
    from torch.optim import Adam
    import random
    
    enc = Seq2SeqEncoder()
    dec = Seq2SeqDecoder()
    criteria = nn.NLLLoss()
    eo = Adam(enc.parameters(), lr=1e-4)
    do = Adam(dec.parameters(), lr=1e-4)
    
    teacher_ratio = 0.5
    
    since = time()
    for inputs, outputs in iterate_dataloader():
        loss = 0
        for x, y in zip(inputs, outputs):
            
            enc_h, enc_c = enc.initHidden()
            out, enc_h, enc_c = enc(x, enc_h, enc_c)
                
            dec_h, dec_c = enc_h.view(3, -1), enc_c.view(3, -1)
            pred = []
            inputw = '<BOS>'
            
            for w in y:
                if w not in WORDS_TO_ID:
                    continue
                out, dec_h, dec_c = dec(inputw, dec_h, dec_c)
                
                if random.random() > teacher_ratio: 
                    prediction = torch.argmax(out.detach(), dim=0)
                    inputw = ID_TO_WORDS[prediction.detach().item()]
                else:
                    inputw = w 
                
                loss += criteria(out.view(1, -1), torch.tensor([WORDS_TO_ID[w]], dtype=torch.long)) / 4
                
            print(loss.item())
        
        eo.zero_grad()
        do.zero_grad()
        loss.backward()
        eo.step()
        do.step()
        
        print(time() - since)
        break
    

    
    
