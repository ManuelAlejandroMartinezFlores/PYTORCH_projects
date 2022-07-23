import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image 
import io

class NeuralNet(nn.Module):
    def __init__(self, input_size=0, hidden_size=500, num_classes=10):
        super(NeuralNet, self).__init__()
        self.conv = nn.Conv2d(1, 3, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(507, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.pool(self.conv(x))
        x = self.flatten(x)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        return self.l3(x)
    
    
model = NeuralNet()
model.load_state_dict(torch.load('app/mnist.pth'))
model.eval()


def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])
    
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image)


def get_prediction(image_tensor):
    images = image_tensor.reshape(1, 1, 28, 28)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    return predicted, nn.functional.softmax(outputs)