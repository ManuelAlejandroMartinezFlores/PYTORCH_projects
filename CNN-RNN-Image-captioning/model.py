from torchvision.models import alexnet, AlexNet_Weights
import torch.nn as nn
import torch.nn.functional as F
from torch_data import N
import torch
import numpy as np



class Encoder(nn.Module):
    def __init__(self, hidden_size=256):
        super(Encoder, self).__init__()
        self.alex = alexnet(weights=AlexNet_Weights.DEFAULT)
        for param in self.alex.parameters():
            param.requires_grad = False
        self.alex.classifier._modules['6'] = nn.Identity()
        self.fc = nn.Linear(4096, hidden_size)
        
    def forward(self, x):
        x = self.alex(x)
        return self.fc(x)
    
    def state_dict(self):
        return self.fc.state_dict()
    
    def load_state_dict(self, dict):
        self.fc.load_state_dict(dict)
        
        
class Decoder(nn.Module):
    def __init__(self, hidden_size=256, output_size=N, num_layers=3, input_size=N):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(hidden_size*2, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def forward(self, x, h, c, img):
        h, c = self.init_hidden()
        x = self.embed(x)
        x = self.dropout(F.relu(x))
        x = torch.cat([x, img], dim=1)
        out, (h, c) = self.lstm(x, (h, c))
        return self.fc(out), h, c 
    
    def init_hidden(self):
        return torch.zeros(self.num_layers, self.hidden_size), torch.zeros(self.num_layers, self.hidden_size)
        


if __name__ == '__main__':
    from torch_data import DATALOADER, sentence2id

    encoder = Encoder()
    decoder = Decoder()
    
    
    criteria = nn.CrossEntropyLoss()
    
    images, captions = iter(DATALOADER).next()
    loss = 0
    for img, cap in zip(images, captions):
        img_out = encoder(img.view(1, 3, 224, 224))
        
        h, c = decoder.init_hidden()
        
        inputw = 0
        for w in sentence2id(cap) + [1]:
            out, h, c = decoder(torch.tensor([inputw]), h, c, img_out)
            outnp = torch.softmax(out, dim=1).detach().numpy().reshape(N)
            inputw = np.random.choice(N, p=outnp)
            
            loss += criteria(out, torch.tensor([w])) / len(cap)
            
    print(loss.item()/32)
    

