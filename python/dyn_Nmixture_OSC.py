#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Binomial, Poisson
from torch.utils.data import DataLoader, TensorDataset
import time
from src.MLP_loss import backward_likelihood, transition_matrix
import gc
import torch.nn.functional as F
import os

recruitment = "absolute" # "absolute" "per_capita"

nsite = 300
nrep = 5
nt = 15
x = torch.distributions.uniform.Uniform(low=-1 * torch.ones(nsite), high=torch.ones(nsite)).sample()
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
    "simulated_data/"
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

####### Define a model ########
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 4)
        self.nt = nt

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        output = self.fc2(x)
        phi = torch.sigmoid(output[:, [0]].repeat(1, self.nt - 1))
        gamma = torch.exp(output[:, [1]].repeat(1, self.nt - 1))
        lambd = torch.exp(output[:, [2]])
        p = torch.sigmoid(output[:, [3]].repeat(1, self.nt))

        return phi, gamma, lambd, p


net = Net()
running_loss = list()

dataset = TensorDataset(torch.tensor(x).float(), torch.tensor(y))
dataloader = DataLoader(dataset, 
                        batch_size=4,#1,#4,#8,#16,#32,#64,#128,#256,
                        shuffle=False, 
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
        x_i, y_i = xy
        # n_max = y_i.numpy().max().astype(np.int32) + 60
        n_max = np.nanmax(y_i).astype(np.int32) + 60
        x_i, y_i = x_i.to(device), y_i.to(device)
        optimizer.zero_grad()
        phi_i, gamma_i, lambda_i, p_i = net(x_i)
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

    phi_hat, gam_hat, lam_hat, p_hat = net(x.to(device))

torch.save(net, f"{save_dir}/net.pth")

torch.cuda.empty_cache()