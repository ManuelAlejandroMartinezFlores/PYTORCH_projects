import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchtext.vocab import GloVe





MAX_DIM = 100
GLOVE = GloVe('6B', MAX_DIM)
len_vectors = GLOVE.vectors.size(0)
GLOVE.stoi['<EOS>'] = len_vectors
GLOVE.stoi['<BOS>'] = len_vectors + 1
GLOVE.stoi['#'] = len_vectors + 2
GLOVE.itos.append('<EOS>')
GLOVE.itos.append('<BOS>') 
GLOVE.itos.append('#')

GLOVE.vectors = torch.cat([GLOVE.vectors, torch.ones(1, MAX_DIM), torch.ones(1, MAX_DIM)*0.5,
                             torch.zeros(1, MAX_DIM)], dim=0)





DATA = torch.load('data/tokenized_data.pth')
X = DATA['x']
LEN_WORDS = DATA['len_words']
PAD = DATA['pad']
MAX_LEN = DATA['max_len']

WORDS_TO_ID = DATA['words_to_id']
ID_TO_WORDS = DATA['id_to_words']

ID_TO_GLOVE = {}
for id, w in ID_TO_WORDS.items():
    ID_TO_GLOVE[id] = GLOVE.stoi[w] if w in GLOVE.stoi else -999
    




class CorpusDataset(Dataset):
    def __init__(self):
        self.x = DATA['x']
        self.y = DATA['y']
        self.n = len(self.x)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n
    
    
dataset = CorpusDataset()
DATALOADER = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
NITERS = len(DATALOADER)


def iterate_dataloader():
    for x, y in DATALOADER:
        xtext = [s.split(' ') for s in x]
        ytext = [s.split(' ') for s in y]
        yield xtext, ytext 
    

if __name__ == '__main__':
    b = GLOVE.get_vecs_by_tokens(X[10].split(' '))
    print(b.shape)
    print(b)
    
    dataset = CorpusDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)
    
    
    cnt = 0
    for xtext, ytext in iterate_dataloader():
        print(xtext, ytext)
        cnt += 1
        if cnt == 2: break
        
    print(ID_TO_GLOVE[1])
    print(len(ID_TO_GLOVE))
        
    
    
    
    
    
    
