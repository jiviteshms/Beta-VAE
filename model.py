import torch
import torch.nn.functional as F
from torch import nn

# input img -> hidden dim -> mean, std_dev -> parametrization trick -> decoder -> output img
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()
        # encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2logvar = nn.Linear(h_dim, z_dim)

        # decoder
        self.z_2hid = nn.Linear(z_dim,h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.img_2hid(x))
        mu, logvar = self.hid_2mu(h), self.hid_2logvar(h)
        return mu, logvar

    def decode(self, z):
        k = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(k))

    def forward(self, x):
        mu, logvar = self.encode(x)
        epsilon = torch.randn_like(mu)
        std = torch.exp(0.5 * logvar)
        z_reparametrized = mu + std * epsilon
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, logvar


if __name__ == "__main__":
    x = torch.randn(4, 28*28) # 4 is the batch size
    vae = VariationalAutoEncoder(input_dim=784)
    x_rec, mu, logvar = vae(x)
    print(x_rec.shape)
    print(mu.shape)
    print(logvar.shape)
    