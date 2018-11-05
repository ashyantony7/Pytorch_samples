import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from SDSAE import AutoEncoder, noise, kl_divergence

# Parameters
numEpochs = 200
batch_size = 16
beta = 1  # for KL 
rho = 0.08  # sparsity constraint
enc_length = 14


# perfrom operations in GPU if possible
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = AutoEncoder(hidden1=20, hidden2=enc_length).to(device)
rho_tensor = torch.Tensor([rho for _ in range(enc_length)]).unsqueeze(0).to(device)

# loss function
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000001,
                             weight_decay=1e-5)

# load checkpoint if available
if os.path.exists('./model_SDSAE.pth'):
    checkpoint = torch.load('./model_SDSAE.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    trained_epoch = checkpoint['trained_epoch']
    print('Model successfully loaded with {} trained epochs'.format(trained_epoch + 1))
else:
    trained_epoch = 0

# fetch the data sets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
dataset = MNIST('../data', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Training Loop
for epoch in range(0, numEpochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        img, _ = data
        img_n = noise(img).to(device)

        # get outputs
        y, enc = model(img_n)

        # loss function
        rho_hat = torch.sum(enc, dim=1, keepdim=True)
        sparse_penalty = beta * kl_divergence(rho_tensor, rho_hat)
        loss = criterion(img, y.to('cpu')) + sparse_penalty.to('cpu')

        # backward
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:  # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                    (trained_epoch + epoch + 1, i + 1, running_loss/100))
            running_loss = 0.0


# save the model
trained_epoch = trained_epoch + epoch + 1
torch.save({'trained_epoch': trained_epoch,
            'model_state_dict': model.state_dict(),
            }, './model_SDSAE.pth')


            

