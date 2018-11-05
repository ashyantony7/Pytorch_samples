import torch
from SDSAE import AutoEncoder, noise
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
enc_length =14
batch_size = 16

# perfrom operations in GPU if possible
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = AutoEncoder(hidden1=20, hidden2=enc_length).to(device)

# load model
checkpoint = torch.load('./model_SDSAE.pth')
model.load_state_dict(checkpoint['model_state_dict'])
trained_epoch = checkpoint['trained_epoch']
print('Model successfully loaded with {} trained epochs'.format(trained_epoch + 1))


def image(img):
    """Function to obtain image from tensor"""
    img = img / 2 + 0.5     # unnormalize
    return img.numpy()

# fetch the data sets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
dataset = MNIST('../data', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# testing with a sample image
dataiter = iter(dataloader)
img_test_set, _ = dataiter.next()
img1 = img_test_set[0, 0, :, :]  # fetch an image

plt.figure(1)
img1 = image(img1)
plt.imshow(img1, cmap='gray')
plt.title('Original Image')


# run through the auto-encoder
out, enc = model(img_test_set.to(device))

# get the tensor back to cpu
out = out.to('cpu')
img2 = out[0, 0, :, :]

plt.figure(2)
img2 = image(img2.detach())
plt.imshow(img2, cmap='gray')
plt.title('Decoded Image')
plt.show()
