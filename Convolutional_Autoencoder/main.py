import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        # encoder layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

        # decoder layers
        self.unpool1 = nn.MaxUnpool2d(2, 2)
        self.convt1 = nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.unpool2 = nn.MaxUnpool2d(2, 2)
        self.convt2 = nn.ConvTranspose2d(in_channels=6, out_channels=1, kernel_size=5, stride=1, padding=0)

    def forward(self, x):
        # encoder
        x = F.relu(self.conv1(x))
        x, ind1 = self.pool1(x)
        x = F.relu(self.conv2(x))
        enc, ind2 = self.pool2(x)
        encoded = enc.view(-1, 16 * 4 * 4)

        # decoder
        x = self.unpool1(enc, ind2)
        x = self.convt1(x)
        x = self.unpool2(x, ind1)
        decoded = self.convt2(x)

        return decoded, encoded


model = autoencoder()

# loss function
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000001,
                             weight_decay=1e-5)

# load checkpoint if available
if os.path.exists('./model.pth'):
    checkpoint = torch.load('./model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print('Model successfully loaded with {} trained epochs'.format(epoch + 1))


# fetch the data sets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = MNIST('./data', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


def image(img):
    """Function to obtain image from tensor"""
    img = img / 2 + 0.5     # unnormalize
    return img.numpy()


# perfrom operations in GPU if possible
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Training Loop
 numepochs = 200
 for epoch in range(0, numepochs):
     running_loss = 0.0
     for i, data in enumerate(dataloader):
         img, _ = data
         img = img.to(device)

         # get outputs
         y, _ = model(img)

         # loss function
         loss = criterion(img, y)

         # backward
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:  # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss/100))
            running_loss = 0.0


# save the model
torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            }, './model.pth')
            

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

