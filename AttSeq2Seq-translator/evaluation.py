import torch 
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd

from model import Seq2SeqEncoder, AttSeq2SeqDecoder 
from torch_data import DATALOADER
from torch_data import en as EN
from torch_data import sp as SP
 
def evaluate(encoder, decoder, filename):

    MAX_LEN = 11
    inputs, outputs = iter(DATALOADER).next()
    
    k = 1
    plt.figure(figsize=(12, 6))
    plt.suptitle('Attention weights')
    
    
    for x, y in zip(inputs, outputs):
        x += ' <EOS>'
        y += ' <EOS>'
        ins = torch.tensor(SP.sentence2ids(x), dtype=torch.long)
        enc_out, h, c = encoder.evaluate(ins)
        enc_out = torch.cat([enc_out, torch.zeros(MAX_LEN - enc_out.size(0), 256)])
        
        attn_weights = []
        pred = [] 
        inputw = 0
        for w in EN.sentence2ids(y):
            out, h, c, attn_w = decoder.evaluate(torch.tensor([inputw], dtype=torch.long), h, c, enc_out)
            attn_weights.append(attn_w[:, :ins.size(0)].view(1, -1).detach())
            inputw = torch.argmax(out, dim=1).detach().item()
            pred.append(EN.id2word[inputw])
            
            
            
        attn_weights = torch.cat(attn_weights, dim=0).detach().numpy()
        
        df = pd.DataFrame(columns=x.split(' '), data=attn_weights)
        df[''] = pred
        df = df.set_index('', drop=True)
        
        plt.subplot(2, 2, k)
        sns.heatmap(df, cmap='bone', vmax=1, vmin=0)
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)

            
        
        
        k += 1
        if k == 5: break
       
    plt.tight_layout() 
    plt.savefig(filename)
    
    
    
    
    
    
def load_model(encoder, decoder):
    try:
        data = torch.load(f'models/AttSeq2Seq.pth')
        encoder.load_state_dict(data['encoder'])
        decoder.load_state_dict(data['decoder'])
        encoder.eval()
        decoder.eval()
        print('loaded')
        print(data['epoch'])
    except:
        pass 
    
    
if __name__ == '__main__':
    encoder = Seq2SeqEncoder()
    decoder = AttSeq2SeqDecoder()
    load_model(encoder, decoder)
    evaluate(encoder, decoder, 'imgs/ev.png')
     
    