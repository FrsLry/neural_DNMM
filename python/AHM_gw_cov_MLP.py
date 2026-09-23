import rdata
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch import nn
from torch.distributions import Binomial, Poisson
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
from python.src.MLP_loss import backward_likelihood, transition_matrix
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Load swiss Green Woodpecker data
gw = rdata.read_rds('R/AHM_data/AHM_green_woodpecker.rds')

# Create data for training
y = gw['C']
y = y.astype(np.float32, copy=False).filled(np.nan)
y = np.swapaxes(y, 1, 2)

nt = int(gw['nyears'])

# Covariates at the site level for lambda
elev = gw['elev'].reshape(-1,1)
forest = gw['forest'].reshape(-1,1)

x_route = np.concatenate(
    [
        elev,
        forest
    ], 
    axis=1
)

# Covaraites at the site-year level for p
date = np.swapaxes(gw['DATE'], 1, 2)
intens = np.swapaxes(gw['INT'], 1, 2)

date = date[...,None]
intens = intens[...,None]


x_p = np.concatenate(
    [
        date,
        intens,
    ], 
    axis=3
)

# Dummy covariate for phi, gamma (no covariate)
x_phi_gamma = np.ones((y.shape[0], 1), dtype=np.float32)

# Create a TensorDataset for training
dataset = TensorDataset(torch.as_tensor(x_route), 
                        torch.as_tensor(x_p), 
                        torch.as_tensor(y),
                        torch.as_tensor(x_phi_gamma))

# Load shared train / validation / test split 
split_df = rdata.read_rds("R/AHM_data/gw_site_split_seed347.rds")

train_idx = split_df.loc[split_df["split"] == "train", "site_python"].to_numpy(dtype=int)
val_idx = split_df.loc[split_df["split"] == "val", "site_python"].to_numpy(dtype=int)
test_idx = split_df.loc[split_df["split"] == "test", "site_python"].to_numpy(dtype=int)

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
test_dataset = Subset(dataset, test_idx)

print("Train sites:", len(train_dataset))
print("Validation sites:", len(val_dataset))
print("Test sites:", len(test_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# Evaluation function for validation loss
def evaluate_loss(model, loader, device, nt):
    model.eval()
    acc_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for x_route_i, x_p_i, y_i, x_phi_gamma_i in loader:
            x_route_i = x_route_i.to(device, non_blocking=True).float()
            x_p_i = x_p_i.to(device, non_blocking=True).float()
            x_phi_gamma_i = x_phi_gamma_i.to(device, non_blocking=True).float()
            y_i = y_i.to(device, non_blocking=True).float()

            safe = torch.nan_to_num(y_i, nan=float('-inf'))
            n_max = int(safe.amax().item()) + 60

            phi_i, gamma_i, lambda_i, p_i = model(x_route_i, x_p_i, x_phi_gamma_i)
            nll = -backward_likelihood(y_i, lambda_i, p_i, phi_i, gamma_i, n_max, nt, recruitment="absolute")
            loss = nll.mean()

            acc_loss += float(loss.item())
            num_batches += 1

    return acc_loss / max(1, num_batches)
###########################


class Net(nn.Module):
    def __init__(self, nt = nt, k_p=x_p.shape[3], x_route_dim = x_route.shape[1]):
        super().__init__()
        self.nt  = nt
        self.k_p = k_p

        # Covariate head for lambda
        self.fc1 = nn.Linear(x_route_dim, 64)
        self.fc2 = nn.Linear(64, 1)

        # No covariate head for phi, gamma
        self.phi_gamma = nn.Linear(1, 2)

        # Detection head
        self.p1 = nn.Linear(k_p, 64)
        self.p2 = nn.Linear(64, 1)

    def forward(self, x_route, x_p, x_phi_gamma):

        # lambda per route
        x = torch.sigmoid(self.fc1(x_route))
        output = self.fc2(x)
        lambd = torch.exp(output[:, [0]])

        # phi, gamma (no covariate)
        out_pg = self.phi_gamma(x_phi_gamma) 
        phi = torch.sigmoid(out_pg[:, [0]]).repeat(1, self.nt - 1)
        gamma = torch.exp(out_pg[:, [1]]).repeat(1, self.nt - 1)

        # detection per (i,t,j)
        xx = torch.sigmoid(self.p1(x_p))
        output_p = self.p2(xx).squeeze(-1)
        p = torch.sigmoid(output_p)  

        return phi, gamma, lambd, p

net = Net()

n_epoch = 500
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-6)
net = net.to(device)

epoch_losses = []
val_losses = []
running_loss = []

## Early stopping class
class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.best_epoch = None
        self.counter = 0
        self.best_state = None
        self.should_stop = False

    def step(self, loss, model, epoch):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_epoch = epoch
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

early_stopper = EarlyStopping(patience=15, min_delta=1e-4)

t_fit_start = time.time()

## Train the model ##
for i in range(n_epoch):
    acc_loss = 0.0
    num_batches = 0
    for i_batch, (x_route_i, x_p_i, y_i, x_phi_gamma_i) in enumerate(train_loader):
        x_route_i = x_route_i.to(device, non_blocking=True).float() 
        x_p_i     = x_p_i.to(device, non_blocking=True).float()     
        x_phi_gamma_i = x_phi_gamma_i.to(device, non_blocking=True).float()  
        y_i       = y_i.to(device, non_blocking=True).float()       

        safe = torch.nan_to_num(y_i, nan=float('-inf'))
        n_max = int(safe.amax().item()) + 60 

        optimizer.zero_grad()
        phi_i, gamma_i, lambda_i, p_i = net(x_route_i, x_p_i, x_phi_gamma_i)
        nll = -backward_likelihood(y_i, lambda_i, p_i, phi_i, gamma_i, n_max, nt)
        loss = nll.mean()
        loss.backward()
        optimizer.step()

        val = float(loss.detach().item())
        acc_loss += val
        num_batches += 1
        running_loss.append(val)

        print('epoch ', i, '/',n_epoch,', batch ', i_batch, ', Maximum abundance: ', n_max)

    epoch_mean = acc_loss / max(1, num_batches)
    epoch_losses.append(epoch_mean)

    val_mean = evaluate_loss(net, val_loader, device, nt)
    val_losses.append(val_mean)
    print(f"Epoch {i+1}/{n_epoch}  train_loss={epoch_mean:.4f}  val_loss={val_mean:.4f}")

    early_stopper.step(val_mean, net, i + 1)

    print(f"   best_val_loss={early_stopper.best_loss:.4f}  patience_counter={early_stopper.counter}")

    if early_stopper.should_stop:
        print(f"Early stopping triggered at epoch {i+1}")
        break

# restore best model
if early_stopper.best_state is not None:
    net.load_state_dict(early_stopper.best_state)
    print(f"Best model restored with loss {early_stopper.best_loss:.4f}")


total_time = time.time() - t_fit_start
print(f"\nTotal training time: {total_time:.2f} sec")

torch.cuda.empty_cache()

# torch.save(net, "python/trained_models/swiss_gw_cov_MLP.pth")

plt.plot(range(1, len(epoch_losses)+1), epoch_losses, marker='o', label="Train")
plt.plot(range(1, len(val_losses)+1), val_losses, marker='o', label="Validation")

plt.xlabel("Epoch")
plt.ylabel("Negative log-likelihood")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

