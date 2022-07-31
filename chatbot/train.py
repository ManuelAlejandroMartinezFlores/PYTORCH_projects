import json
from model import NeuralNet
from nltk_utils import stem, tokenize, bag_of_words
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


with open('intents.json', 'r') as f:
    intents = json.load(f)
    
ignore_words = ['?', '!', '.', '.']

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        words = tokenize(pattern)
        all_words.extend([stem(w) for w in words if w not in ignore_words])
        xy.append((words, tag))
        

all_words = sorted(set(all_words))
tags = sorted(set(tags))


x_train = []
y_train = []

for pattern_sentence, tag in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
    
    label = tags.index(tag)
    y_train.append(label)
    
x_train = np.array(x_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train 
        self.y_data = y_train 
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples 
        
        

  
         
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=0)


model = NeuralNet(len(all_words), 8, len(tags))


criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    for epoch in range(1000):
        for words, labels in train_loader:
            outputs = model(words)
            loss = criteria(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch % 100 == 99:
            print(f'epoch {epoch+1:4d}/1000,  loss: {loss.item():.6f}')

    data = {
        'model_state': model.state_dict(),
        'input_size': len(all_words),
        'output_size': len(tags),
        'hidden_size': 8,
        'all_words': all_words,
        'tags': tags
    }

    FILE = 'data.pth'
    torch.save(data, FILE)

    print(f'Data saved to: {FILE}')
    
if __name__ == '__main__':
    train()