import torch 
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

from model import EncoderDecoder

from torch_data import TESTLOADER


def load():
    model = EncoderDecoder()
    try:
        data = torch.load('models/pix2pix.pth')
        model.load_state_dict(data['generator'])
        return model 
    except:
        return model 
    
    
def visualize_result(segmentations, photos, predictions, filename):
    k = 1
    trans = ToPILImage()
    plt.figure()
    for seg, photo, pred in zip(segmentations, photos, predictions):
        plt.subplot(2, 3, k)
        plt.imshow(trans(seg))
        plt.axis('off')
        plt.title('Segmentation')
        
        plt.subplot(2, 3, k+1)
        plt.imshow(trans(photo))
        plt.axis('off')
        plt.title('Ground Truth')
        
        plt.subplot(2, 3, k+2)
        plt.imshow(trans(pred))
        plt.axis('off')
        plt.title('Prediction')
        
        k += 3
        if k >= 6: break
        
    plt.tight_layout()
    plt.savefig(filename)
    
def evaluation_test(filename):
    segmentations, photos = iter(TESTLOADER).next()
    
    model = load()
    with torch.no_grad():
        model.eval()
        predictions = model(segmentations)
        
        visualize_result(segmentations, photos, predictions, filename)
    
    
if __name__ == '__main__':
    from torch_data import TESTLOADER
    
    segmentations, photos = iter(TESTLOADER).next()
    
    model = load()
    with torch.no_grad():
        model.eval()
        predictions = model(segmentations)
        
        visualize_result(segmentations, photos, predictions, 'imgs/ev.png')