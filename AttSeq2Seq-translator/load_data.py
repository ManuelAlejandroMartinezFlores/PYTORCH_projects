import re 
import torch 


def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()
    
    text = re.sub(r"i'm", "i am", text) 
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|]", "", text)
    text = re.sub(r"\!", " !", text)
    text = re.sub(r"\?", " ?", text)
    text = re.sub(r"\¡", "¡ ", text)
    text = re.sub(r"\¿", "¿ ", text)
    text = re.sub(r"\.", " .", text)
    text = re.sub(r"\,", " ,", text)
    
    return text

def tokenize(text):
    return [w for w in clean_text(text).split(' ') if w != "" and " " not in w]


class Lang:
    def __init__(self, lang):
        self.lang = lang 
        self.word2id = {'<BOS>': 0, '<EOS>': 1, '<UNK>': 2}
        self.id2word = {0: '<BOS>', 1: '<EOS>', 2: '<UNK>'}
        self.word2count = {'<BOS>': 0, '<EOS>': 0, '<UNK>': 0}
        self.n_words = 3
        self.sentences = []
        
    def load(self):
        DATA = torch.load(f'data/{self.lang}.pth')
        self.__load_dict(DATA)
        
    def __load_dict(self, dict):
        self.word2id = dict['word2id']
        self.id2word = dict['id2word']
        self.word2count = dict['word2count']
        self.sentences = dict['sentences']
        self.n_words = len(self.word2id)
        
        
    def add_sentences(self, sentence, tokens):
        self.sentences.append(sentence)
        for word in tokens:
            if word in self.word2count:
                self.word2count[word] += 1
            else:
                self.word2count[word] = 1 
                
            
    def sentence2ids(self, sentence):
        ans = []
        for w in tokenize(sentence):
            if w in self.word2id:
                ans.append(self.word2id[w])
            else:
                ans.append(self.word2id['<UNK>'])
        return ans 
    
    def ids2sentence(self, ids):
        return' '.join([self.id2word[i] for i in ids])
            
    
    def _conclude(self, min_ap=3):
        for w, c in self.word2count.items():
            if c > min_ap:
                self.word2id[w] = self.n_words
                self.id2word[self.n_words] = w
                self.n_words += 1
            
        
    def save(self):
        DATA = {
            'word2id': self.word2id,
            'id2word': self.id2word,
            'word2count': self.word2count,
            'sentences': self.sentences
        }
        torch.save(DATA, f'data/{self.lang}.pth')
            
        

if __name__ == '__main__':
    en = Lang('en')
    sp = Lang('sp')
    
    print(clean_text("¿Hola? Es una, prueba."))
    
    with open('data/spa.txt', 'r') as f:
        for line in f.readlines():
            line = line.split('\t')[:2]
            clean_en = clean_text(line[0])
            tok_en = tokenize(clean_en)
            clean_sp = clean_text(line[1])
            tok_sp = tokenize(clean_sp)
            
            if len(tok_sp) > 15 or len(tok_en) > 15:
                continue 
            en.add_sentences(clean_en, tok_en) 
            sp.add_sentences(clean_sp, tok_sp)
            
            
    en._conclude()
    sp._conclude()
    en.save()
    sp.save()
    
    print(en.n_words, sp.n_words)
    print(len(en.word2id))
    
