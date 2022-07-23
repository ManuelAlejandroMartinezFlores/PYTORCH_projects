import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


input_size = 784
hidden_size = 500
num_classes = 10 
nepochs = 6
batch_size = 100
lr = 0.001

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])


train_dataset = torchvision.datasets.MNIST(root="./data", train=True,
                                           transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False,
                                           transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                           shuffle=False)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.conv = nn.Conv2d(1, 3, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(507, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        return self.l3(x)
    
    
model = NeuralNet(input_size, hidden_size, num_classes)

criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

n_total_steps = len(train_loader)
best_acc = 0

for epoch in range(nepochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(100, -1, 28, 28)
        
        pred = model(images)
        loss = criteria(pred, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 99:   
            with torch.no_grad():
                correct = 0
                samples = 0 
                for images, labels in test_loader:
                    images = images.reshape(100, -1, 28, 28)
                    pred = model(images)
                    
                    _, predictions = torch.max(pred, 1)
                    samples += labels.shape[0]
                    correct += (predictions == labels).sum().item()
                    
                if best_acc < correct/samples:
                    best_acc = correct/samples
                    torch.save(model.state_dict(), 'app.mnist.pth')
            print(f'epoch {epoch+1} / {nepochs}, step {i+1:4d}/{n_total_steps}, loss {loss.item():.5f}, best acc {best_acc:.4f}')
            
            
print(f'Best acc: {best_acc:.4f}')
            
