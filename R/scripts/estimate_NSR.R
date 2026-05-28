library(ggplot2)
library(reticulate)
library(tidyr)
library(dplyr)
library(ggh4x)
use_condaenv("C:/Users/leroy.64/.conda/envs/py3_12_11/", required = TRUE)

torch <- import("torch")
nn <- import("torch.nn")
np <- import("numpy")
source_python("../python/src/MLP_loss.py")

nt <- 15

x <- seq(-1, 1, length.out = 100)
x_tensor <- torch$tensor(matrix(x, ncol = 1), dtype = torch$float32)

device <- torch$device(
  if (torch$cuda$is_available()) "cuda" else "cpu"
)

py_run_string(sprintf("
import torch
import torch.nn as nn

nt = %d

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
", nt))


dirs <- list.dirs("../simulated_data/")[-1]

predictions_NN <- list()
predictions_jags <- list()

latent_NN   <- list()
latent_jags <- list()

x_values <- list()

for(dir in dirs){

  sim_id <- basename(dir)

  ### Load simulation-specific x and y ####
  yPath <- normalizePath(file.path(dir, "y.npy"))
  xPath <- normalizePath(file.path(dir, "x.npy"))

  y <- np$load(yPath)
  x_sim <- np$load(xPath)

  x_values[[sim_id]] <- x_sim

  y_tensor <- torch$from_numpy(y)
  x_tensor <- torch$from_numpy(x_sim)$float()$unsqueeze(1L)

  ## Neural Net ####
  model_path <- file.path(dir, "net.pth")

  # load model
  model <- torch$load(model_path, weights_only = FALSE, map_location = device)
  model$eval()

  with(torch$no_grad(), {
    pred <- model(x_tensor$to(device))

    phi_hat    <- pred[[1]]
    gamma_hat  <- pred[[2]]
    lambda_hat <- pred[[3]]
    p_hat      <- pred[[4]]

    res <- estimating_NSR(
      y_tensor$to(device),
      lambda_hat,
      p_hat,
      phi_hat,
      gamma_hat,
      batch_size = 16L
    )

  })

  latent_NN[[sim_id]] <- list(
    E_N = res[[1]]$detach()$cpu()$numpy(),
    E_S = res[[2]]$detach()$cpu()$numpy(),
    E_R = res[[3]]$detach()$cpu()$numpy()
  )

}

# saveRDS(latent_NN, "latent_NN.rds")


#### Estimates form Computer Vision
py_run_string("
import torch

def make_images(x_input, img_h=16, img_w=16, noise_sd=0.0):
    img_c = 1

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, img_h),
        torch.linspace(-1, 1, img_w),
        indexing='ij'
    )

    top_pattern    = 1 - (yy + 1) / 2
    bottom_pattern = (yy + 1) / 2
    center_pattern = torch.exp(-(yy**2) / 0.18)

    images_out = torch.zeros((len(x_input), img_c, img_h, img_w))

    for i in range(len(x_input)):
        xi = float(x_input[i].item())

        w_center = 1 - abs(xi)
        w_top    = max(-xi, 0.0)
        w_bottom = max(xi, 0.0)

        w_sum = w_top + w_center + w_bottom
        w_top /= w_sum
        w_center /= w_sum
        w_bottom /= w_sum

        img = (
            w_top * top_pattern +
            w_center * center_pattern +
            w_bottom * bottom_pattern
        )

        if noise_sd > 0:
            img = img + noise_sd * torch.randn_like(img)

        img = torch.clamp(img, 0.0, 1.0)
        images_out[i, 0] = img

    return images_out
")


py_run_string(sprintf("
import torch
import torch.nn as nn

nt = %d

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
", nt))

dirs <- list.dirs("../simulated_data_CNN/")[-1]

predictions_NN <- list()

latent_NN   <- list()

x_values <- list()

for(dir in dirs){

  sim_id <- basename(dir)

  ### Load simulation-specific x and y ####
  yPath <- normalizePath(file.path(dir, "y.npy"))
  xPath <- normalizePath(file.path(dir, "x.npy"))

  y <- np$load(yPath)
  x_sim <- np$load(xPath)

  x_values[[sim_id]] <- x_sim

  y_tensor <- torch$from_numpy(y)
  x_tensor <- torch$from_numpy(x_sim)$float()$unsqueeze(1L)

  images_tensor <- py$make_images(
    x_tensor,
    img_h = 16L,
    img_w = 16L,
    noise_sd = 0.0
  )

  ## Neural Net ####
  model_path <- file.path(dir, "net.pth")

  # load model
  model <- torch$load(model_path, weights_only = FALSE, map_location = device)
  model$eval()

  with(torch$no_grad(), {
    pred <- model(images_tensor$to(device))

    phi_hat    <- pred[[1]]
    gamma_hat  <- pred[[2]]
    lambda_hat <- pred[[3]]
    p_hat      <- pred[[4]]

    res <- estimating_NSR(
      y_tensor$to(device),
      lambda_hat,
      p_hat,
      phi_hat,
      gamma_hat,
      batch_size = 16L
    )

  })

  latent_NN[[sim_id]] <- list(
    E_N = res[[1]]$detach()$cpu()$numpy(),
    E_S = res[[2]]$detach()$cpu()$numpy(),
    E_R = res[[3]]$detach()$cpu()$numpy()
  )

}

# saveRDS(latent_NN, "latent_CNN.rds")

