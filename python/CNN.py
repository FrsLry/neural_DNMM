#!/usr/bin/env python

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from src.MLP_loss import backward_likelihood
import time
import os

nsite = 300
nrep = 5
nt = 15

recruitment = "absolute" # "absolute" "per_capita"

x = torch.distributions.uniform.Uniform(
    low=-1 * torch.ones(nsite),
    high=torch.ones(nsite)
).sample()
x, _ = torch.sort(x)
x = x.unsqueeze(1)

# draw coefficients for gamma/lambda
g0, g1, g2 = np.random.uniform(-2, 2, 3)
l0, l1, l2 = np.random.uniform(-2, 2, 3)

gamma_rate = torch.exp(g0 + g1 * x + g2 * x**2)
lambda_rate = torch.exp(l0 + l1 * x + l2 * x**2)

# draw coefficients for detection/survival on logit scale for more varied shapes
p0, p1, p2 = np.random.uniform(-2.5, 2.5, 3)
s0, s1, s2 = np.random.uniform(-2.5, 2.5, 3)

pr_detection = torch.sigmoid(p0 + p1 * x + p2 * x**2)
pr_survival  = torch.sigmoid(s0 + s1 * x + s2 * x**2)

### Generate true abundance, survival and recruitment ##########################################################
n0 = torch.poisson(lambda_rate).squeeze()

## Simulate TRUE abundance, survival and recruitment
## Create the empty tensors
n = torch.zeros(nsite, nt)
r = torch.zeros(nsite, nt - 1)
s = torch.zeros(nsite, nt - 1)

# Add the abundance at time 0
n[:,0] = torch.tensor(n0).squeeze()

## To simulate survivors, iterate over each time step because phi depends on abundance at t-1
for t in range(1, nt):
    s[:, t-1] = torch.tensor(np.random.binomial(
        n=n[:, t-1].cpu().numpy().astype(int),
        p=pr_survival.squeeze().cpu().numpy()
    ))
    r[:, t-1] = torch.poisson(gamma_rate).squeeze()
    # r[:, t-1] = torch.poisson(n[:, t-1].unsqueeze(1) * gamma_rate).squeeze() # if per capita recruitment
    n[:, t] = s[:, t-1] + r[:, t-1]


## Simulate observed abundance over repeated surveys ##########################################################
y = np.random.binomial(n=n[...,None],
                       p=torch.tensor(pr_detection).numpy()[...,None],
                       size=(nsite, nt, nrep))

# Build a readable run folder with all coefficients
save_dir = (
    "simulated_data_CNN/"
    f"g_{g0:.3f}_{g1:.3f}_{g2:.3f}"
    f"__l_{l0:.3f}_{l1:.3f}_{l2:.3f}"
    f"__p_{p0:.3f}_{p1:.3f}_{p2:.3f}"
    f"__s_{s0:.3f}_{s1:.3f}_{s2:.3f}"
)
os.makedirs(save_dir, exist_ok=True)

np.save(f"{save_dir}/x.npy", x.squeeze(1).numpy())
np.save(f"{save_dir}/y.npy", y)
np.save(f"{save_dir}/n.npy", n)
np.save(f"{save_dir}/s.npy", s)
np.save(f"{save_dir}/r.npy", r)

### Generating images from x for CNN input ##########################################################
img_h = 16
img_w = 16
img_c = 1

yy, xx = torch.meshgrid(
    torch.linspace(-1, 1, img_h),
    torch.linspace(-1, 1, img_w),
    indexing="ij"
)

# basis patterns
top_pattern    = 1 - (yy + 1) / 2          # white at top, black at bottom
bottom_pattern = (yy + 1) / 2              # black at top, white at bottom
center_pattern = torch.exp(-(yy**2) / 0.18)  # bright horizontal band in middle

images = torch.zeros((nsite, img_c, img_h, img_w))

for i in range(nsite):
    xi = float(x[i].item())   # in [-1, 1]

    # weights
    w_center = 1 - abs(xi)
    w_top    = max(-xi, 0.0)
    w_bottom = max(xi, 0.0)

    # normalize weights so they sum to 1
    w_sum = w_top + w_center + w_bottom
    w_top /= w_sum
    w_center /= w_sum
    w_bottom /= w_sum

    img = (
        w_top * top_pattern +
        w_center * center_pattern +
        w_bottom * bottom_pattern
    )

    # small noise 
    # img = img + 0.05 * torch.randn_like(img)

    # clamp only
    img = torch.clamp(img, 0.0, 1.0)

    images[i, 0] = img


class Net(nn.Module):
    def __init__(self, nt):
        super().__init__()
        self.nt = nt
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            # nn.Softplus(),
            nn.Sigmoid(),
            nn.AvgPool2d(2),   # 16 -> 8
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            # nn.Softplus(),            
            nn.Sigmoid(),
            nn.AvgPool2d(2)    # 8 -> 4
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 64),
            nn.Sigmoid(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        z = self.features(x)
        out = self.fc(z)

        phi   = torch.sigmoid(out[:, [0]].repeat(1, self.nt - 1))
        gamma = torch.exp(out[:, [1]].repeat(1, self.nt - 1))
        lambd = torch.exp(out[:, [2]])
        p     = torch.sigmoid(out[:, [3]].repeat(1, self.nt))

        return phi, gamma, lambd, p
    

net = Net(nt=nt)
running_loss = list()

dataset = TensorDataset(images.float(), torch.tensor(y).float())
dataloader = DataLoader(dataset, 
                        batch_size=32,#1,#4,#8,#16,#32,#64,#128,#256,
                        shuffle=True, 
                        num_workers=0,
                        pin_memory=True)

n_epoch = 300
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-6)
running_loss = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = net.to(device)

epoch_losses = []
running_loss = []

## Train the model ##
for i in range(n_epoch):
    t0 = time.time()
    acc_loss = 0.0
    num_batches = 0
    for i_batch, xy in enumerate(dataloader):
        img_i, y_i = xy
        n_max = np.nanmax(y_i).astype(np.int32) + 60
        img_i, y_i = img_i.to(device), y_i.to(device)
        optimizer.zero_grad()
        phi_i, gamma_i, lambda_i, p_i = net(img_i)
        nll = -backward_likelihood(y_i, lambda_i, p_i, phi_i, gamma_i, n_max, nt, recruitment=recruitment) # greedy
        loss = torch.mean(nll)        
        loss.backward()
        optimizer.step() # Does the update
        print('epoch ', i, '/',n_epoch,', batch ', i_batch, ', Maximum abundance: ', n_max)

        val = float(loss.detach().item())
        acc_loss += val
        num_batches += 1
        running_loss.append(val)

    print('{} seconds'.format(time.time() - t0))
    epoch_mean = acc_loss / max(1, num_batches)
    epoch_losses.append(epoch_mean)
    print(f"Epoch {i+1}/{n_epoch}  train_loss={epoch_mean:.4f}")

    net.eval()
    with torch.no_grad():
        phi_hat, gam_hat, lam_hat, p_hat = net(images.to(device))
    net.train()

torch.save(net, f"{save_dir}/net.pth")

torch.cuda.empty_cache()