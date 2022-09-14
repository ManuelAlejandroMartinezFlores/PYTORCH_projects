
import nltk 
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
import torch
import re


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
    
    return text

def tokenize(sentence:str):
    sentence = clean_text(sentence)
    return  nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word)

def padding(tokenized_sentence:list, max_len:int, pad:str):
    if len(tokenized_sentence) > max_len-1:
        tokenized_sentence = tokenized_sentence[:max_len-1]
    tokenized_sentence = (max_len - len(tokenized_sentence)) * [pad] + tokenized_sentence
    return tokenized_sentence

MAX_LEN = 10

def reduce_len(sentence, max_len=MAX_LEN):
    if len(sentence) > MAX_LEN:
        sentence = sentence[:15]
    return sentence


if __name__ == '__main__':

    from convokit import Corpus, download


    corpus = Corpus(filename='./data/movie-corpus') 

    DATA = {'x': [], 'y': []}
    WORDS_TO_ID = {}
    ID_TO_WORDS = {}
    WORDS_FREQ = {}
    
    


    for utter in corpus.iter_utterances(lambda x: x.reply_to is not None):
        ytok = tokenize(utter.text) + ['<EOS>']
        
        
        reply_to = corpus.get_utterance(utter.reply_to)
        xtok = tokenize(reply_to.text) + ['<EOS>']
        
        
        if len(ytok) > MAX_LEN + 1 or len(xtok) > MAX_LEN + 1:
            continue
        
        for w in ytok: 
            if w not in WORDS_FREQ:
                WORDS_FREQ[w] = 1 
            else:
                WORDS_FREQ[w] += 1
                
        DATA['x'].append(' '.join(xtok))
        DATA['y'].append(' '.join(ytok))         
                
        for w in xtok: 
            if w not in WORDS_FREQ:
                WORDS_FREQ[w] = 1 
            else:
                WORDS_FREQ[w] += 1
        if len(WORDS_FREQ) >= 20000: 
            break
        
    cnt = 0
      
    for w in WORDS_FREQ:
        if WORDS_FREQ[w] > 5:
            WORDS_TO_ID[w] = cnt
            ID_TO_WORDS[cnt] = w 
            cnt += 1
        
    new_x = []
    new_y = []
    
    for x, y in zip(DATA['x'], DATA['y']):
        flag = True
        for w in x.split(' '):
            if w not in WORDS_TO_ID:
                flag = False 
        for w in y.split(' '):
            if w not in WORDS_TO_ID:
                flag = False 
                
        if flag:
            new_x.append(x)
            new_y.append(y)
        
    DATA['x'] = new_x 
    DATA['y'] = new_y

    DATA['words_to_id'] = WORDS_TO_ID
    DATA['id_to_words'] = ID_TO_WORDS
    DATA['len_words'] = cnt
    DATA['pad'] = '#'
    DATA['max_len'] = MAX_LEN + 1

    torch.save(DATA, 'data/tokenized_data.pth')
    print(DATA['x'][10], DATA['y'][10])
    print(len(DATA['x']))
    print(cnt+1)



        
        



