import os
import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn, optim
from model import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 500  
Z_DIM = 40   
NUM_EPOCHS = 20  
BATCH_SIZE = 128  
LR_RATE = 1e-3   
BETA = 0.6 

# create images folder
os.makedirs("images", exist_ok=True)

# dataset
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
loss_fn = nn.BCELoss(reduction="sum")

# training
for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))
    for i,(x,y) in loop:
        # forward pass
        x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
        x_reconstructed, mu, logvar = model(x)

        # compute loss
        reconstruction_loss = loss_fn(x_reconstructed, x)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # backpropagation
        loss = reconstruction_loss + BETA * kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

# inference
model = model.to("cpu")

def inference(digit, num_examples=1):
    images = []
    idx = 0

    for x,y in dataset:
        if y==idx:
            images.append(x)
            idx += 1

        if idx == 10:
            break

    encodings_digit = []

    for d in range(10):
        with torch.no_grad():
            mu, logvar = model.encode(images[d].view(1,784))

        encodings_digit.append((mu,logvar))

    mu, logvar = encodings_digit[digit]

    std = torch.exp(0.5 * logvar)

    for example in range(num_examples):

        epsilon = torch.randn_like(std)

        z = mu + std * epsilon

        out = model.decode(z)

        out = out.view(-1, 1, 28, 28)

        save_image(out, f"generated_images/generated_{digit}_ex{example}.png")

for idx in range(10):
    inference(idx, num_examples=5)