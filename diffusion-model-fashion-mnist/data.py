import torch 
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader

batch_size = 24

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t * 2 - 1)
])

train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True,
                                           transform=trans)
REAL_LOADER = DataLoader(dataset=train_dataset, batch_size=32,
                                           shuffle=True)


if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    plt.figure()
    
    for k, (photo, _) in enumerate(REAL_LOADER):
        if k == 4: break 
        plt.subplot(2, 2, k+1)
        plt.imshow(transforms.ToPILImage()(photo[0]), cmap='gray')
        plt.axis('off')
        
    plt.savefig('imgs/data.png')