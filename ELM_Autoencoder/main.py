import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class elm_ae(nn.Module):
    def __init__(self, hidden=100):
        super(elm_ae, self).__init__()
        self.fc1 = nn.Linear(28, hidden)  

    def forward(self, x):
        # hidden layer matrix
        H = torch.sigmoid(self.fc1(x))

        # intialize weights and output matrix
        B = torch.zeros(H.transpose(2, 3).shape).to(device).detach()
        X = torch.zeros(x.shape).to(device).detach()

        # batch iterative operation
        for i in range(0, H.shape[0]):

            # calculate the weight matrix
            B[i, 0, :, :] = torch.mm(torch.pinverse(H[i, 0, :, :]), x[i, 0, :, :])
            
            # get the outputs
            X[i, 0, :, :] = torch.mm(H[i, 0, :, :], B[i, 0, :, :])
            
        return X, B


# get the model
model = elm_ae(hidden=10).to(device)

# fetch the data sets
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = MNIST('./data', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


def image(img):
    """Function to obtain image from tensor"""
    img = img / 2 + 0.5     # unnormalize
    return img.numpy()


dataiter = iter(dataloader)
img_test_set, _ = dataiter.next()
img1 = img_test_set[0, 0, :, :]  # fetch an image


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

# show the decoded image
plt.figure(2)
img2 = image(img2.detach())
plt.imshow(img2, cmap='gray')
plt.title('Decoded Image')
plt.show()


