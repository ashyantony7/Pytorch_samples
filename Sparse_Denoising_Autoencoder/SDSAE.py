import torch
from torch import nn


def noise(image, level=0.05):
    """Adds noise to an image tensor
     level is of range (0, 1)"""

    idx = torch.rand(image.shape)
    image[idx < level] = -1.0
    image[idx > (1 - level)] = 1.0
    return image


def kl_divergence(p, q):
    """calculates kl divergence between two tensors"""

    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2


class AutoEncoder(nn.Module):
    def __init__(self, hidden1=20, hidden2=14):
        super(AutoEncoder, self).__init__()
        # encoder layers
        self.h1 = nn.Linear(28, hidden1)
        self.h2 = nn.Linear(hidden1, hidden2)

        # decoder layers
        self.h2_d = nn.Linear(hidden2, hidden1)
        self.h1_d = nn.Linear(hidden1, 28)

    def forward(self, x):
        # encoder
        x = torch.sigmoid(self.h1(x))
        encoded = torch.sigmoid(self.h2(x))

        # decoder
        decoded = self.h2_d(encoded)
        decoded = self.h1_d(decoded)

        return decoded, encoded
