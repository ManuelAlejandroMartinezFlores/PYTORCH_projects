from torchvision.models import alexnet, AlexNet_Weights
import torch.nn as nn
import torch.nn.functional as F
from torch_data import N
import torch



class Encoder(nn.Module):
    def __init__(self, hidden_size=256):
        super(Encoder, self).__init__()
        self.alex = alexnet(weights=AlexNet_Weights.DEFAULT)
        for param in self.alex.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(1000, hidden_size * 6)
        
    def forward(self, x):
        x = F.relu(self.alex(x))
        return self.fc(x)
    
    def state_dict(self):
        return self.fc.state_dict()
    
    def load_state_dict(self, dict):
        self.fc.load_state_dict(dict)
        
        
class Decoder(nn.Module):
    def __init__(self, hidden_size=256, output_size=N, num_layers=3, input_size=N):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h, c):
        x = self.embed(x)
        out, (h, c) = self.lstm(x, (h, c))
        return torch.log_softmax(self.fc(out), dim=1), h, c 
        


if __name__ == '__main__':
    from torch_data import DATALOADER, sentence2id

    encoder = Encoder()
    decoder = Decoder()
    
    criteria = nn.NLLLoss()
    
    images, captions = iter(DATALOADER).next()
    loss = 0
    for img, cap in zip(images, captions):
        out = encoder(img.view(1, 3, 224, 224)).view(6, 256)
        h = out[:3]
        c = out[3:]
        
        inputw = 0
        for w in sentence2id(cap) + [1]:
            out, h, c = decoder(torch.tensor([inputw]), h, c)
            inputw = torch.argmax(out, dim=1)
            
            loss += criteria(out, torch.tensor([w])) / len(cap)
            
    print(loss.item())
    

