import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from load_data import Lang


class LangData(Dataset):
    def __init__(self, lang_from:Lang, lang_to:Lang):
        self.x = lang_from.sentences 
        self.y = lang_to.sentences 
        self.n = len(self.x)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n
    
    
    
en = Lang('en')
en.load()
sp = Lang('sp')
sp.load()

N = {
    'en': en.n_words,
    'sp': sp.n_words
}


dataset = LangData(sp, en)
DATALOADER = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

def iterate_dataloader():
    for x, y in DATALOADER:
        x = [sp.sentence2ids(s) + [1] for s in x]
        y = [en.sentence2ids(s) + [1] for s in y]
        yield x, y
        
        
def ids_to_sentence(ids, lang):
    if lang == 'sp':
        return sp.ids2sentence(ids)
    else:
        return en.ids2sentence(ids)
    
        
        
if __name__ == '__main__':
    
    MAX_LEN = 0
    MAX_EN = 0
    MAX_SP = 0
    for x, y in iterate_dataloader():
        for i, j in zip(x, y):
            if max(i) > MAX_SP:
                MAX_SP = max(i)
            if max(j) > MAX_EN:
                MAX_EN = max(j)
            if len(i) > MAX_LEN:
                MAX_LEN = len(i)
            if len(j) > MAX_LEN:
                MAX_LEN = len(j)
                
    print(MAX_LEN)
    print(MAX_SP, sp.n_words)
    print(MAX_EN, en.n_words)
    assert MAX_SP <= sp.n_words
    assert MAX_EN <= en.n_words
    
    
                
    
    