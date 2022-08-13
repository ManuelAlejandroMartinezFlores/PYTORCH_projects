import torch 
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset 
import matplotlib.pyplot as plt
from torchvision.models import AlexNet_Weights
import re


alex_trans = AlexNet_Weights.DEFAULT.transforms()


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
    

TRAIN = datasets.CocoCaptions('./data/imgs', './data/captions_val2017.json')


    
    
# TRAINLOADER = DataLoader(TRAIN, batch_size=1, shuffle=True)


if __name__ == '__main__': 
    
    # images, captions = iter(TRAINLOADER).next()
    
    k = 1
    plt.figure()
    while k < 5:
        img, cap = TRAIN[k]
        print(cap)
        plt.subplot(2, 2, k)
        plt.imshow(img)
        plt.axis('off')
        
        plt.subplot(2, 2, k+1)
        img = alex_trans(img).detach().numpy()
        print(img.shape)
        plt.imshow(img.reshape(224, 224, 3))
        plt.axis('off')
        
        k += 2
        if k == 5: break
    plt.savefig('imgs/transforms.png')
    
    
    cnt = 2
    WORDS2ID = {'<BOS>': 0, '<EOS>': 1}
    ID2WORS = {0: '<BOS>', 1: '<EOS>'}
    for k in range(len(TRAIN)):
        _, cap = TRAIN[k]
        for s in cap:
            for w in tokenize(s):
                if w not in WORDS2ID:
                    WORDS2ID[w] = cnt 
                    ID2WORS[cnt] = w 
                    cnt += 1 
                    
    DATA = {'w2i': WORDS2ID, 'i2w': ID2WORS}
    torch.save(DATA, 'data/dicts.pth')
    print(cnt)