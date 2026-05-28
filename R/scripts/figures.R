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

latent_NN   <- readRDS("latent_NN.rds")
latent_jags <- list()

x_values <- list()

for(dir in dirs){

  sim_id <- basename(dir)

  ### Load simulation-specific x ####
  xPath <- normalizePath(file.path(dir, "x.npy"))
  x_sim <- np$load(xPath)
  x_values[[sim_id]] <- x_sim
  x_tensor <- torch$from_numpy(x_sim)$float()$unsqueeze(1L)

  ## Neural Net ####
  model_path <- file.path(dir, "net.pth")

  # load model
  model <- torch$load(model_path, weights_only = FALSE, map_location = device)
  model$eval()

  with(torch$no_grad(), {
    pred <- model(x_tensor$to(device))
  })

  predictions_NN[[basename(dir)]] <- list(
    phi   = pred[[1]]$detach()$cpu()$numpy()[,1],
    gamma = pred[[2]]$detach()$cpu()$numpy()[,1],
    lambd = pred[[3]]$detach()$cpu()$numpy()[,1],
    p     = pred[[4]]$detach()$cpu()$numpy()[,1]
  )

  #### JAGS ####
  out <- readRDS(file.path(dir, "jagsOut.rds"))

  alpha.lam <- out$mean$alpha.lam
  beta.lam  <- out$mean$beta.lam
  beta.lam.square <- out$mean$beta.lam.square

  alpha.phi <- out$mean$alpha.phi
  beta.phi  <- out$mean$beta.phi
  beta.phi.square <- out$mean$beta.phi.square

  alpha.gamma <- out$mean$alpha.gamma
  beta.gamma  <- out$mean$beta.gamma
  beta.gamma.square <- out$mean$beta.gamma.square

  alpha.p <- out$mean$alpha.p
  beta.p  <- out$mean$beta.p
  beta.p.square <- out$mean$beta.p.square

  predictions_jags[[basename(dir)]] <- list(
     lambd = exp(alpha.lam + beta.lam * x_sim + beta.lam.square * x_sim^2),
     phi   = plogis(alpha.phi + beta.phi * x_sim + beta.phi.square * x_sim^2),
     gamma = exp(alpha.gamma + beta.gamma * x_sim + beta.gamma.square * x_sim^2),
     p     = plogis(alpha.p + beta.p * x_sim + beta.p.square * x_sim^2)
   )

  latent_jags[[sim_id]] <- list(
    E_N = out$mean$N,
    E_S = out$mean$S,
    E_R = out$mean$R
  )

}


predictions_sim <- list()

for (dir in dirs) {

  sim_id <- basename(dir)
  x_sim <- x_values[[sim_id]]

  parts <- strsplit(sim_id, "__", fixed = TRUE)[[1]]

  g <- as.numeric(strsplit(parts[grep("^g_", parts)], "_")[[1]][-1])
  l <- as.numeric(strsplit(parts[grep("^l_", parts)], "_")[[1]][-1])
  pcoef <- as.numeric(strsplit(parts[grep("^p_", parts)], "_")[[1]][-1])
  s <- as.numeric(strsplit(parts[grep("^s_", parts)], "_")[[1]][-1])

  predictions_sim[[sim_id]] <- list(
    gamma = exp(g[1] + g[2] * x_sim + g[3] * x_sim^2),
    lambd = exp(l[1] + l[2] * x_sim + l[3] * x_sim^2),
    p     = plogis(pcoef[1] + pcoef[2] * x_sim + pcoef[3] * x_sim^2),
    phi   = plogis(s[1] + s[2] * x_sim + s[3] * x_sim^2)
  )
}

## create data format for plot
preds <- data.frame(value = as.numeric(),
                    x = as.numeric(),
                    param = as.character(),
                    sim_id= as.character(),
                    type = as.character())


params <- c("phi", "gamma", "lambd", "p")

### Neural Net ###
for(i in seq_along(predictions_NN)){

  sim_name <- names(predictions_NN)[i]
  x_sim <- x_values[[sim_name]]

  for(param in params){

    preds <- rbind(
      preds,
      data.frame(
        value = predictions_NN[[i]][[param]],
        x = x_sim,
        param = param,
        sim_id = sim_name,
        type = "NN"
      )
    )
  }
}

### JAGS ###
for(i in seq_along(predictions_jags)){

  sim_name <- names(predictions_jags)[i]
  x_sim <- x_values[[sim_name]]

  for(param in params){

    preds <- rbind(
      preds,
      data.frame(
        value = predictions_jags[[i]][[param]],
        x = x_sim,
        param = param,
        sim_id = sim_name,
        type = "JAGS"
      )
    )
  }
}

### Simulated ###
for(i in seq_along(predictions_sim)){

  sim_name <- names(predictions_sim)[i]
  x_sim <- x_values[[sim_name]]

  for(param in params){

    preds <- rbind(
      preds,
      data.frame(
        value = predictions_sim[[i]][[param]],
        x = x_sim,
        param = param,
        sim_id = sim_name,
        type = "Simulated"
      )
    )
  }
}

#### Making figure ####
pdf("../figures/sim_comparisons.pdf", height = 5.83, width = 8.27)


preds %>%
  filter(sim_id %in% unique(preds$sim_id)) %>%
  mutate(
    param = case_when(
      param == "phi"   ~ "phi~' (survival)'",
      param == "gamma" ~ "gamma~'(recruitment)'",
      param == "lambd" ~ "lambda~'(abundance '~t[1]*')'",
      param == "p"     ~ "p~'(detection)'"
    ),
    type = case_when(
      type == "JAGS"      ~ "Bayesian",
      type == "NN"        ~ "Neural Network",
      type == "Simulated" ~ "Simulation"
    )
  ) %>%
  ggplot() +
  geom_line(aes(x, value, colour = sim_id, linetype = type), linewidth = 0.7) +
  scale_linetype_manual(
    values = c(
      "Simulation" = "solid",
      "Bayesian" = "dotted",
      "Neural Network" = "dashed"
    )
  ) +
  facet_wrap(vars(param), scales = "free", labeller = label_parsed) +
  facetted_pos_scales(
    y = list(
      param %in% c(
        "phi~' (survival)'",
        "p~'(detection)'"
      ) ~ scale_y_continuous(limits = c(0, 1)),
      param %in% c(
        "gamma~'(recruitment)'"
      ) ~ scale_y_log10(),
      param %in% c(
        "lambda~'(abundance '~t[1]*')'"
      ) ~ scale_y_continuous(trans = "sqrt")
    )
  ) +
  ylab("Parameter value") +
  xlab("Simulated covariate (x)") +
  guides(
    colour = "none",
    linetype = guide_legend(nrow = 1)
  ) +
  theme(
    panel.background = element_blank(),
    panel.grid.major = element_line(color = "grey85", size = 0.3),
    panel.grid.minor = element_blank(),
    axis.line = element_line(color = "black"),
    strip.background = element_blank(),
    strip.text = element_text(size = 10),
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 8),
    legend.key.height = unit(0.4, "lines"),
    legend.key.width  = unit(1.2, "lines"),
    legend.spacing.x = unit(0.3, "lines"),
    legend.box.margin = margin(-5, 0, -5, 0),
    panel.spacing = unit(0.8, "lines")
  )

dev.off()


## N, S, R plots
#### N ###########
df_compare <- data.frame(
  sim_id = character(),
  N_NN = numeric(),
  N_Bayes = numeric(),
  N_Sim = numeric()
)


for (sim_id in names(latent_NN)) {

  N_nn <- as.numeric(latent_NN[[sim_id]]$E_N)
  N_bayes <- as.numeric(latent_jags[[sim_id]]$E_N)
  N_sim <- as.vector(np$load(file.path("../simulated_data",sim_id, "n.npy")))

  tmp <- data.frame(
    sim_id = sim_id,
    N_NN = N_nn,
    N_Bayes = N_bayes,
    N_Sim = N_sim
  )

  df_compare <- rbind(df_compare, tmp)
}


rmse_nn <- df_compare %>%
  group_by(sim_id) %>%
  summarise(rmse = sqrt(mean((N_Sim - N_NN)^2, na.rm = TRUE))) %>%
  summarise(mean_rmse = mean(rmse)) %>%
  pull(mean_rmse)

rmse_bayes <- df_compare %>%
  group_by(sim_id) %>%
  summarise(rmse = sqrt(mean((N_Sim - N_Bayes)^2, na.rm = TRUE))) %>%
  summarise(mean_rmse = mean(rmse)) %>%
  pull(mean_rmse)

lab_nn <- paste0("Mean RMSE = ", round(rmse_nn, 2))
lab_bayes <- paste0("Mean RMSE = ", round(rmse_bayes, 2))

p1_N <-
df_compare %>%
  select(-"N_Bayes") %>%
  ggplot(aes(x = N_Sim,
             y = N_NN,
             colour = sim_id)) +
  ggrastr::rasterise(geom_point(alpha = 0.5, show.legend = F), dpi = 300)+
  geom_function(
    fun = function(x) x,
    linetype = "dashed",
    colour = "black"
  ) +
  annotate("text", x = 1, y = max(df_compare$N_Bayes + 1, na.rm = TRUE),
           label = lab_nn, hjust = 0, vjust = 1, size = 4)+
  theme_bw() +
  labs(
    x = "N (Simulation)",
    y = "N (Neural Network)",
    colour = "Simulation"
  )#+
  # scale_x_log10()+
  # scale_y_log10()

p2_N <-
df_compare %>%
  select(-"N_NN") %>%
  ggplot(aes(x = N_Sim,
             y = N_Bayes,
             colour = sim_id)) +
  ggrastr::rasterise(geom_point(alpha = 0.5, show.legend = F), dpi = 300)+
  geom_function(
    fun = function(x) x,
    linetype = "dashed",
    colour = "black"
  ) +
  annotate("text", x = 1, y = max(df_compare$N_Bayes + 1, na.rm = TRUE),
           label = lab_bayes, hjust = 0, vjust = 1, size = 4)+
  theme_bw() +
  labs(
    x = "N (Simulation)",
    y = "N (Bayesian)",
    colour = "Simulation"
  )#+
  # scale_x_log10()+
  # scale_y_log10()

p3_N <-
df_compare %>%
  select(-"N_Sim") %>%
  ggplot(aes(x = N_NN,
             y = N_Bayes,
             colour = sim_id)) +
  ggrastr::rasterise(geom_point(alpha = 0.5, show.legend = F), dpi = 300)+
  geom_function(
    fun = function(x) x,
    linetype = "dashed",
    colour = "black"
  ) +
  theme_bw() +
  labs(
    x = "N (Neural Network)",
    y = "N (Bayesian)",
    colour = "Simulation"
  )#+
  # scale_x_log10()+
  # scale_y_log10()

pdf("../figures/N_comparisons.pdf", height = 5.83, width = 15)

cowplot::plot_grid(p1_N, p2_N, p3_N, labels = c('A', 'B', 'C'), label_size = 12, nrow = 1)

dev.off()


#### S ##########
df_compare <- data.frame(
  sim_id = character(),
  S_NN = numeric(),
  S_Bayes = numeric(),
  S_Sim = numeric()
)

for (sim_id in names(latent_NN)) {

  S_nn <- as.numeric(latent_NN[[sim_id]]$E_S)
  S_bayes <- na.omit(as.numeric(latent_jags[[sim_id]]$E_S))
  S_sim <- as.vector(np$load(file.path("../simulated_data",sim_id, "s.npy")))

  tmp <- data.frame(
    sim_id = sim_id,
    S_NN = S_nn,
    S_Bayes = S_bayes,
    S_Sim = S_sim
  )

  df_compare <- rbind(df_compare, tmp)
}


rmse_nn <- df_compare %>%
  group_by(sim_id) %>%
  summarise(rmse = sqrt(mean((S_Sim - S_NN)^2, na.rm = TRUE))) %>%
  summarise(mean_rmse = mean(rmse)) %>%
  pull(mean_rmse)

rmse_bayes <- df_compare %>%
  group_by(sim_id) %>%
  summarise(rmse = sqrt(mean((S_Sim - S_Bayes)^2, na.rm = TRUE))) %>%
  summarise(mean_rmse = mean(rmse)) %>%
  pull(mean_rmse)

lab_nn <- paste0(" Mean RMSE = ", round(rmse_nn, 2))
lab_bayes <- paste0("Mean RMSE = ", round(rmse_bayes, 2))

p1_S <-
  df_compare %>%
  select(-"S_Bayes") %>%
  ggplot(aes(x = S_Sim,
             y = S_NN,
             colour = sim_id)) +
  ggrastr::rasterise(geom_point(alpha = 0.5, show.legend = F), dpi = 300) +
  geom_function(
    fun = function(x) x,
    linetype = "dashed",
    colour = "black"
  ) +
  annotate("text", x = 1, y = max(df_compare$S_Bayes + 1, na.rm = TRUE),
           label = lab_nn, hjust = 0, vjust = 1, size = 4)+
  theme_bw() +
  labs(
    x = "S (Simulation)",
    y = "S (Neural Network)",
    colour = "Simulation"
  )#+
  # scale_x_log10()+
  # scale_y_log10()

p2_S <-
  df_compare %>%
  select(-"S_NN") %>%
  ggplot(aes(x = S_Sim,
             y = S_Bayes,
             colour = sim_id)) +
  ggrastr::rasterise(geom_point(alpha = 0.5, show.legend = F), dpi = 300) +
  geom_function(
    fun = function(x) x,
    linetype = "dashed",
    colour = "black"
  ) +
  annotate("text", x = 1, y = max(df_compare$S_Bayes + 1, na.rm = TRUE),
           label = lab_bayes, hjust = 0, vjust = 1, size = 4)+
  theme_bw() +
  labs(
    x = "S (Simulation)",
    y = "S (Bayesian)",
    colour = "Simulation"
  )#+
  # scale_x_log10()+
  # scale_y_log10()

p3_S <-
  df_compare %>%
  select(-"S_Sim") %>%
  ggplot(aes(x = S_NN,
             y = S_Bayes,
             colour = sim_id)) +
  ggrastr::rasterise(geom_point(alpha = 0.5, show.legend = F), dpi = 300) +
  geom_function(
    fun = function(x) x,
    linetype = "dashed",
    colour = "black"
  ) +
  theme_bw() +
  labs(
    x = "S (Neural Network)",
    y = "S (Bayesian)",
    colour = "Simulation"
  )#+
  # scale_x_log10()+
  # scale_y_log10()

pdf("../figures/S_comparisons.pdf", height = 5.83, width = 15)

cowplot::plot_grid(p1_S, p2_S, p3_S, labels = c('A', 'B', 'C'), label_size = 12, nrow = 1)

dev.off()


#### R ##########
df_compare <- data.frame(
  sim_id = character(),
  R_NN = numeric(),
  R_Bayes = numeric(),
  R_Sim = numeric()
)

for (sim_id in names(latent_NN)) {

  R_nn <- as.numeric(latent_NN[[sim_id]]$E_R)
  R_bayes <- na.omit(as.numeric(latent_jags[[sim_id]]$E_R))
  R_sim <- as.vector(np$load(file.path("../simulated_data",sim_id, "r.npy")))

  tmp <- data.frame(
    sim_id = sim_id,
    R_NN = R_nn,
    R_Bayes = R_bayes,
    R_Sim = R_sim
  )

  df_compare <- rbind(df_compare, tmp)
}


rmse_nn <- df_compare %>%
  group_by(sim_id) %>%
  summarise(rmse = sqrt(mean((R_Sim - R_NN)^2, na.rm = TRUE))) %>%
  summarise(mean_rmse = mean(rmse)) %>%
  pull(mean_rmse)

rmse_bayes <- df_compare %>%
  group_by(sim_id) %>%
  summarise(rmse = sqrt(mean((R_Sim - R_Bayes)^2, na.rm = TRUE))) %>%
  summarise(mean_rmse = mean(rmse)) %>%
  pull(mean_rmse)

lab_nn <- paste0("Mean RMSE = ", round(rmse_nn, 2))
lab_bayes <- paste0("Mean RMSE = ", round(rmse_bayes, 2))

p1_R <-
  df_compare %>%
  select(-"R_Bayes") %>%
  ggplot(aes(x = R_Sim,
             y = R_NN,
             colour = sim_id)) +
  ggrastr::rasterise(geom_point(alpha = 0.5, show.legend = F), dpi = 300)+
  geom_function(
    fun = function(x) x,
    linetype = "dashed",
    colour = "black"
  ) +
  annotate("text", x = 1, y = max(df_compare$R_Bayes + 1, na.rm = TRUE),
           label = lab_nn, hjust = 0, vjust = 1, size = 4)+
  theme_bw() +
  labs(
    x = "R (Simulation)",
    y = "R (Neural Network)",
    colour = "Simulation"
  )#+
  # scale_x_log10()+
  # scale_y_log10()

p2_R <-
  df_compare %>%
  select(-"R_NN") %>%
  ggplot(aes(x = R_Sim,
             y = R_Bayes,
             colour = sim_id)) +
  ggrastr::rasterise(geom_point(alpha = 0.5, show.legend = F), dpi = 300)+
  geom_function(
    fun = function(x) x,
    linetype = "dashed",
    colour = "black"
  ) +
  annotate("text", x = 1, y = max(df_compare$R_Bayes + 1, na.rm = TRUE),
           label = lab_bayes, hjust = 0, vjust = 1, size = 4)+
  theme_bw() +
  labs(
    x = "R (Simulation)",
    y = "R (Bayesian)",
    colour = "Simulation"
  )#+
  # scale_x_log10()+
  # scale_y_log10()

p3_R <-
  df_compare %>%
  select(-"R_Sim") %>%
  ggplot(aes(x = R_NN,
             y = R_Bayes,
             colour = sim_id)) +
  ggrastr::rasterise(geom_point(alpha = 0.5, show.legend = F), dpi = 300)+
  geom_function(
    fun = function(x) x,
    linetype = "dashed",
    colour = "black"
  ) +
  theme_bw() +
  labs(
    x = "R (Neural Network)",
    y = "R (Bayesian)",
    colour = "Simulation"
  )#+
  # scale_x_log10()+
  # scale_y_log10()

pdf("../figures/R_comparisons.pdf", height = 5.83, width = 15)

cowplot::plot_grid(p1_R, p2_R, p3_R, labels = c('A', 'B', 'C'), label_size = 12, nrow = 1)

dev.off()

#### All together ####
pdf("../figures/NSR_comparisons.pdf", height = 8.74, width = 8.27)

cowplot::plot_grid(p1_N, p2_N,
                   p1_S, p2_S,
                   p1_R, p2_R, nrow = 3)

dev.off()

#### Supplementary Fig.: NN vs Bayesian estimates  ####
pdf("../figures/FigS1.pdf", height = 5.38, width = 10.16)

cowplot::plot_grid(p3_N, p3_S, p3_R, nrow = 1)

dev.off()


## COmputer vision part: #########################
## CNN
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
            nn.Sigmoid(),
            nn.AvgPool2d(2),   # 16 -> 8
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
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

## Function ot simulate images for predictions with CNN
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


dirs <- list.dirs("../simulated_data_CNN/")[-1]

predictions_NN <- list()

latent_NN   <- readRDS("latent_CNN.rds")

x_values <- list()

for(dir in dirs){

  sim_id <- basename(dir)

  ### Load simulation-specific x ####
  xPath <- normalizePath(file.path(dir, "x.npy"))
  x_sim <- np$load(xPath)
  x_values[[sim_id]] <- x_sim
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
  })

  predictions_NN[[basename(dir)]] <- list(
    phi   = pred[[1]]$detach()$cpu()$numpy()[,1],
    gamma = pred[[2]]$detach()$cpu()$numpy()[,1],
    lambd = pred[[3]]$detach()$cpu()$numpy()[,1],
    p     = pred[[4]]$detach()$cpu()$numpy()[,1]
  )

}


predictions_sim <- list()

for (dir in dirs) {

  sim_id <- basename(dir)
  x_sim <- x_values[[sim_id]]

  parts <- strsplit(sim_id, "__", fixed = TRUE)[[1]]

  g <- as.numeric(strsplit(parts[grep("^g_", parts)], "_")[[1]][-1])
  l <- as.numeric(strsplit(parts[grep("^l_", parts)], "_")[[1]][-1])
  pcoef <- as.numeric(strsplit(parts[grep("^p_", parts)], "_")[[1]][-1])
  s <- as.numeric(strsplit(parts[grep("^s_", parts)], "_")[[1]][-1])

  predictions_sim[[sim_id]] <- list(
    gamma = exp(g[1] + g[2] * x_sim + g[3] * x_sim^2),
    lambd = exp(l[1] + l[2] * x_sim + l[3] * x_sim^2),
    p     = plogis(pcoef[1] + pcoef[2] * x_sim + pcoef[3] * x_sim^2),
    phi   = plogis(s[1] + s[2] * x_sim + s[3] * x_sim^2)
  )
}

## create data format for CNN plot
preds <- data.frame(value = as.numeric(),
                    x = as.numeric(),
                    param = as.character(),
                    sim_id = as.character(),
                    type = as.character())

params <- c("phi", "gamma", "lambd", "p")

### Neural Net ###
for(i in seq_along(predictions_NN)){

  sim_name <- names(predictions_NN)[i]
  x_sim <- x_values[[sim_name]]

  for(param in params){

    preds <- rbind(
      preds,
      data.frame(
        value = predictions_NN[[i]][[param]],
        x = x_sim,
        param = param,
        sim_id = sim_name,
        type = "NN"
      )
    )
  }
}

### Simulated ###
for(i in seq_along(predictions_sim)){

  sim_name <- names(predictions_sim)[i]
  x_sim <- x_values[[sim_name]]

  for(param in params){

    preds <- rbind(
      preds,
      data.frame(
        value = predictions_sim[[i]][[param]],
        x = x_sim,
        param = param,
        sim_id = sim_name,
        type = "Simulated"
      )
    )
  }
}

## Plot
pdf("../figures/CNN_comparisons.pdf", height = 5.83, width = 8.27)

preds %>%
  filter(sim_id %in% unique(preds$sim_id)) %>%
  mutate(
    param = case_when(
      param == "phi"   ~ "phi~' (survival)'",
      param == "gamma" ~ "gamma~'(recruitment)'",
      param == "lambd" ~ "lambda~'(abundance '~t[1]*')'",
      param == "p"     ~ "p~'(detection)'"
    ),
    type = case_when(
      type == "NN"        ~ "Neural Network",
      type == "Simulated" ~ "Simulation"
    )
  ) %>%
  ggplot() +
  geom_line(aes(x, value, colour = sim_id, linetype = type), linewidth = 0.7) +
  scale_linetype_manual(
    values = c(
      "Simulation" = "solid",
      "Neural Network" = "dashed"
    )
  ) +
  facet_wrap(vars(param), scales = "free", labeller = label_parsed) +
  facetted_pos_scales(
    y = list(
      param %in% c(
        "phi~' (survival)'",
        "p~'(detection)'"
      ) ~ scale_y_continuous(limits = c(0, 1)),
      param %in% c(
        "gamma~'(recruitment)'"
      ) ~ scale_y_log10(),
      param %in% c(
        "lambda~'(abundance '~t[1]*')'"
      ) ~ scale_y_continuous(trans = "sqrt")
    )
  ) +
  ylab("Parameter value") +
  xlab("Simulated images") +
  guides(
    colour = "none",
    linetype = guide_legend(nrow = 1)
  ) +
  theme(
    panel.background = element_blank(),
    panel.grid.major = element_line(color = "grey85", size = 0.3),
    panel.grid.minor = element_blank(),
    axis.line = element_line(color = "black"),
    strip.background = element_blank(),
    strip.text = element_text(size = 10),
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 8),
    legend.key.height = unit(0.4, "lines"),
    legend.key.width  = unit(1.2, "lines"),
    legend.spacing.x = unit(0.3, "lines"),
    legend.box.margin = margin(-5, 0, -5, 0),
    axis.text.x = element_blank(),
    panel.spacing = unit(0.8, "lines")
  )

dev.off()



## N, S, R plots
df_compare_all <- data.frame()

for (sim_id in names(latent_NN)[-5]) {

  tmp <- bind_rows(
    data.frame(
      sim_id = sim_id,
      metric = "N",
      NN = as.numeric(latent_NN[[sim_id]]$E_N),
      Simulation = as.vector(np$load(file.path("../simulated_data_CNN/", sim_id, "n.npy")))
    ),
    data.frame(
      sim_id = sim_id,
      metric = "S",
      NN = as.numeric(latent_NN[[sim_id]]$E_S),
      Simulation = as.vector(np$load(file.path("../simulated_data_CNN/", sim_id, "s.npy")))
    ),
    data.frame(
      sim_id = sim_id,
      metric = "R",
      NN = as.numeric(latent_NN[[sim_id]]$E_R),
      Simulation = as.vector(np$load(file.path("../simulated_data_CNN/", sim_id, "r.npy")))
    )
  )

  df_compare_all <- bind_rows(df_compare_all, tmp)
}

rmse_labs <- df_compare_all %>%
  group_by(metric, sim_id) %>%
  summarise(
    rmse = sqrt(mean((Simulation - NN)^2, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  group_by(metric) %>%
  summarise(
    lab = paste0("Mean RMSE = ", round(mean(rmse, na.rm = TRUE), 2)),
    x = min(df_compare_all$Simulation, na.rm = TRUE),
    y = max(df_compare_all$NN, na.rm = TRUE),
    .groups = "drop"
  )


pdf("../figures/CNN_NSR.pdf", height = 4, width = 8.27)

df_compare_all %>%
  ggplot(aes(x = Simulation + 1, y = NN + 1)) +
  ggrastr::rasterise(
    geom_point(aes(colour = sim_id), alpha = 0.5, show.legend = FALSE),
    dpi = 300
  ) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed",
    #colour = "red",
    linewidth = 0.7
  ) +
  geom_text(
    data = rmse_labs,
    aes(x = x + 1, y = y + 1, label = lab),
    inherit.aes = FALSE,
    hjust = 0,
    vjust = 1,
    size = 3.5
  ) +
  facet_wrap(~ metric, scales = "free") +
  #scale_x_log10() +
  #scale_y_log10() +
  labs(
    x = "Simulation",
    y = "Neural Network"
  ) +
  theme(
    panel.background = element_blank(),
    panel.grid.major = element_line(color = "grey85", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    axis.line = element_line(color = "black"),
    strip.background = element_blank(),
    strip.text = element_text(size = 10),
    panel.spacing = unit(0.8, "lines"),
    legend.position = "none"
  )

dev.off()

### Figures Swiss green woodpecker #####
#### NN ######
gw_data_jags <- readRDS("AHM_data/AHM_green_woodpecker.rds")

nt <- gw_data_jags$nyears

## Rebuild the covariates for loading the NN
y <- gw_data_jags$C
y <- as.array(y)

## Reshape observed data
# y is site x rep x year and I want site x year x rep:
y <- aperm(y, c(1, 3, 2))

# route-level covariates for lambda
elev <- matrix(gw_data_jags$elev, ncol = 1)
forest <- matrix(gw_data_jags$forest, ncol = 1)

x_route <- cbind(elev, forest)

# site-year-rep covariates for p
date <- aperm(as.array(gw_data_jags$DATE), c(1, 3, 2))
intens <- aperm(as.array(gw_data_jags$INT), c(1, 3, 2))

# add last dimension = number of detection covariates
x_p <- array(NA_real_, dim = c(dim(date)[1], dim(date)[2], dim(date)[3], 2))
x_p[,,,1] <- date
x_p[,,,2] <- intens

# dummy covariate for phi/gamma
x_phi_gamma <- matrix(1, nrow = dim(y)[1], ncol = 1)
x_route_dim <- as.integer(ncol(x_route))
k_p <- as.integer(dim(x_p)[4])

py_run_string(sprintf("
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, nt = %d, k_p=%d, x_route_dim = %d):
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
", nt, k_p, x_route_dim))

model <- torch$load("../python/trained_models/swiss_gw_cov_MLP.pth", weights_only = FALSE, map_location = "cpu")

model$eval()

## Function to predict lambda relationships fixing the other covaraites to their means
predict_lambda_nn <- function(x_route_grid, model, nt, nrep, k_p) {
  B <- as.integer(nrow(x_route_grid))

  x_route_t <- torch$tensor(x_route_grid, dtype = torch$float32)
  x_phi_gamma_t <- torch$ones(as.integer(B), as.integer(1), dtype = torch$float32)
  x_p_dummy_t <- torch$zeros(
    as.integer(B), as.integer(nt), as.integer(nrep), as.integer(k_p),
    dtype = torch$float32
  )

  with(torch$no_grad(), {
    pred <- model(x_route_t, x_p_dummy_t, x_phi_gamma_t)
  })

  py_to_r(pred[[3]]$detach()$numpy())[, 1]
}

## Function to predict p relationships fixing the other covaraites to their means
predict_p_nn <- function(x_p_grid, model, x_route_dim) {
  B <- as.integer(dim(x_p_grid)[1])

  x_p_t <- torch$tensor(x_p_grid, dtype = torch$float32)
  x_route_dummy_t <- torch$zeros(as.integer(B), as.integer(x_route_dim), dtype = torch$float32)
  x_phi_gamma_t <- torch$ones(as.integer(B), as.integer(1), dtype = torch$float32)

  with(torch$no_grad(), {
    pred <- model(x_route_dummy_t, x_p_t, x_phi_gamma_t)
  })

  py_to_r(pred[[4]]$detach()$numpy())[, 1, 1]
}

## Set arguments and mean covariates
source(file="AHM_data/AHM2_02.02.R")
nt <- as.integer(nt)
nrep <- as.integer(dim(x_p)[3])
k_p <- as.integer(dim(x_p)[4])
x_route_dim <- as.integer(ncol(x_route))

forest_mean <- mean(forest)
elev_mean <- mean(elev)

elev.grid <- seq(min(peckers$elev),max(peckers$elev),,100)
x.elev <- standardize2match(elev.grid, peckers$elev)

forest.grid <- seq(min(peckers$forest),max(peckers$forest),,100)
x.fore <- standardize2match(forest.grid, peckers$forest)

## Predict lambda relationships with elevation
x_route_elev <- cbind(
  x.elev,
  rep(forest_mean, length(elev.grid))
)

lam_nn_elev <- predict_lambda_nn(
  x_route_grid = x_route_elev,
  model = model,
  nt = nt,
  nrep = nrep,
  k_p = k_p
)

## Predict lambda relationships with forest cover
x_route_forest <- cbind(
  rep(elev_mean, length(forest.grid)),
  x.fore
)


lam_nn_forest <- predict_lambda_nn(
  x_route_grid = x_route_forest,
  model = model,
  nt = nt,
  nrep = nrep,
  k_p = k_p
)


date_mean <- mean(date, na.rm = TRUE)
int_mean  <- mean(intens, na.rm = TRUE)

date.grid <- seq(min(date, na.rm = TRUE), max(date, na.rm = TRUE), , 100)
int.grid  <- seq(min(intens, na.rm = TRUE), max(intens, na.rm = TRUE), , 100)

# Predict p relationships with DATE
x_p_date <- array(0, dim = c(length(date.grid), 1, 1, 2))
x_p_date[, 1, 1, 1] <- date.grid
x_p_date[, 1, 1, 2] <- int_mean

p_nn_date <- predict_p_nn(
  x_p_grid = x_p_date,
  model = model,
  x_route_dim = x_route_dim
)

## Predict p relationships with INT
x_p_int <- array(0, dim = c(length(int.grid), 1, 1, 2))
x_p_int[, 1, 1, 1] <- date_mean
x_p_int[, 1, 1, 2] <- int.grid

p_nn_int <- predict_p_nn(
  x_p_grid = x_p_int,
  model = model,
  x_route_dim = x_route_dim
)


## JAGS ######
library(AHMbook)

out <- readRDS("AHM_data/jagsOut_gw_covariates.rds")

inv_logit <- function(x) 1 / (1 + exp(-x))

post <- out$sims.list

# lambda ~ elevation
lam_draws_elev <- sapply(seq_along(elev.grid), function(i) {
  exp(
    post$alpha.lam +
      post$beta.elev  * x.elev[i] +
      post$beta.elev2 * x.elev[i]^2 +
      post$beta.for   * 0
  )
})

lam_q_elev <- t(apply(lam_draws_elev, 2, quantile, probs = c(0.025, 0.5, 0.975)))
colnames(lam_q_elev) <- c("lower", "value", "upper")

# lambda ~ forest
lam_draws_forest <- sapply(seq_along(forest.grid), function(i) {
  exp(
    post$alpha.lam +
      post$beta.elev  * 0 +
      post$beta.elev2 * 0 +
      post$beta.for   * x.fore[i]
  )
})

lam_q_forest <- t(apply(lam_draws_forest, 2, quantile, probs = c(0.025, 0.5, 0.975)))
colnames(lam_q_forest) <- c("lower", "value", "upper")

# p ~ date
p_draws_date <- sapply(seq_along(date.grid), function(i) {
  inv_logit(
    post$alpha.p +
      post$beta.jul  * date.grid[i] +
      post$beta.jul2 * date.grid[i]^2 +
      post$beta.int  * int_mean +
      post$beta.int2 * int_mean^2
  )
})

p_q_date <- t(apply(p_draws_date, 2, quantile, probs = c(0.025, 0.5, 0.975)))
colnames(p_q_date) <- c("lower", "value", "upper")

# p ~ intensity
p_draws_int <- sapply(seq_along(int.grid), function(i) {
  inv_logit(
    post$alpha.p +
      post$beta.jul  * date_mean +
      post$beta.jul2 * date_mean^2 +
      post$beta.int  * int.grid[i] +
      post$beta.int2 * int.grid[i]^2
  )
})

p_q_int <- t(apply(p_draws_int, 2, quantile, probs = c(0.025, 0.5, 0.975)))
colnames(p_q_int) <- c("lower", "value", "upper")

ribbons_gw <- bind_rows(
  data.frame(
    x = elev.grid,
    lower = lam_q_elev[, "lower"],
    value = lam_q_elev[, "value"],
    upper = lam_q_elev[, "upper"],
    param = "lambda",
    type = "Bayesian",
    covariate = "elevation"
  ),
  data.frame(
    x = forest.grid,
    lower = lam_q_forest[, "lower"],
    value = lam_q_forest[, "value"],
    upper = lam_q_forest[, "upper"],
    param = "lambda",
    type = "Bayesian",
    covariate = "forest"
  ),
  data.frame(
    x = date.grid,
    lower = p_q_date[, "lower"],
    value = p_q_date[, "value"],
    upper = p_q_date[, "upper"],
    param = "p",
    type = "Bayesian",
    covariate = "date"
  ),
  data.frame(
    x = int.grid,
    lower = p_q_int[, "lower"],
    value = p_q_int[, "value"],
    upper = p_q_int[, "upper"],
    param = "p",
    type = "Bayesian",
    covariate = "intensity"
  )
) %>%
  mutate(
    param = case_when(
      param == "lambda" ~ "lambda~'(abundance '~t[1]*')'",
      param == "p" ~ "p~'(detection)'",
      TRUE ~ param
    ),
    covariate = case_when(
      covariate == "elevation" ~ "Elevation (m)",
      covariate == "forest" ~ "Forest cover (%)",
      covariate == "date" ~ "Julian date",
      covariate == "intensity" ~ "Survey intensity",
      TRUE ~ covariate
    )
  )

preds_gw <- bind_rows(
  data.frame(value = lam_nn_elev, x = elev.grid, param = "lambda", type = "NN", covariate = "elevation"),
  data.frame(value = lam_nn_forest, x = forest.grid, param = "lambda", type = "NN", covariate = "forest"),
  data.frame(value = p_nn_date, x = date.grid, param = "p", type = "NN", covariate = "date"),
  data.frame(value = p_nn_int, x = int.grid, param = "p", type = "NN", covariate = "intensity"),

  data.frame(value = lam_q_elev[, "value"], x = elev.grid, param = "lambda", type = "Bayesian", covariate = "elevation"),
  data.frame(value = lam_q_forest[, "value"], x = forest.grid, param = "lambda", type = "Bayesian", covariate = "forest"),
  data.frame(value = p_q_date[, "value"], x = date.grid, param = "p", type = "Bayesian", covariate = "date"),
  data.frame(value = p_q_int[, "value"], x = int.grid, param = "p", type = "Bayesian", covariate = "intensity")
) %>%
  mutate(
    param = case_when(
      param == "lambda" ~ "lambda~'(abundance '~t[1]*')'",
      param == "p" ~ "p~'(detection)'",
      TRUE ~ param
    ),
    covariate = case_when(
      covariate == "elevation" ~ "Elevation (m)",
      covariate == "forest" ~ "Forest cover (%)",
      covariate == "date" ~ "Julian date",
      covariate == "intensity" ~ "Survey intensity",
      TRUE ~ covariate
    ),
    type = case_when(
      type == "NN" ~ "Neural Network",
      type == "Bayesian" ~ "Bayesian",
      TRUE ~ type
    )
  )


# pdf("../figures/gw_comparisons.pdf", height = 6.58, width = 7.05)
pdf("../figures/gw_comparisons.pdf", height = 5.83, width = 8.27)

ggplot() +
  geom_ribbon(
    data = ribbons_gw,
    aes(x = x, ymin = lower, ymax = upper),
    fill = "grey70",
    alpha = 0.25
  ) +
  geom_line(
    data = preds_gw,
    aes(x = x, y = value, linetype = type, colour = type),
    linewidth = 0.7
  ) +
  scale_linetype_manual(
    values = c(
      "Bayesian" = "solid",
      "Neural Network" = "longdash"
    )
  ) +
  facet_wrap(
    vars(param, covariate),
    scales = "free",
    labeller = labeller(param = label_parsed)
  ) +
  ylab("Parameter value") +
  xlab("Covariate") +
  theme(
    panel.background = element_blank(),
    panel.grid.major = element_line(color = "grey85", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    axis.line = element_line(color = "black"),

    strip.background = element_blank(),
    strip.text = element_text(size = 10),

    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 8),
    legend.key.height = unit(0.4, "lines"),
    legend.key.width = unit(1.2, "lines"),
    legend.spacing.x = unit(0.3, "lines"),
    legend.box.margin = margin(-5, 0, -5, 0),

    panel.spacing = unit(0.8, "lines")
  )

dev.off()


## Estimate N, S, R
# For NN
x_route_t <- torch$tensor(x_route, dtype = torch$float32)
x_p_t <- torch$tensor(x_p, dtype = torch$float32)
x_phi_gamma_t <- torch$tensor(x_phi_gamma, dtype = torch$float32)
y <- array(as.numeric(y), dim = dim(y))
y_t <- torch$tensor(y, dtype = torch$float32)

with(torch$no_grad(), {
  pred <- model(x_route_t, x_p_t, x_phi_gamma_t)
})

phi_hat_t <- pred[[1]]
gamma_hat_t <- pred[[2]]
lambda_hat_t <- pred[[3]]
p_hat_t <- pred[[4]]

res <- estimating_NSR(
  y_t,
  lambda_hat_t,
  p_hat_t,
  phi_hat_t,
  gamma_hat_t,
  batch_size = 16L
)

N_nn = res[[1]]$detach()$cpu()$numpy()
S_nn = res[[2]]$detach()$cpu()$numpy()
R_nn = res[[3]]$detach()$cpu()$numpy()

N_bayes <- out$mean$N
S_bayes <- out$mean$S[,-1]
R_bayes <- out$mean$R[,-1]

df_compare <- bind_rows(
  data.frame(
    NN = as.vector(N_nn),
    JAGS = as.vector(N_bayes),
    metric = "N"
  ),
  data.frame(
    NN = as.vector(S_nn),
    JAGS = as.vector(S_bayes),
    metric = "S"
  ),
  data.frame(
    NN = as.vector(R_nn),
    JAGS = as.vector(R_bayes),
    metric = "R"
  )
)

pdf("../figures/greenWoodpecker_NSR_comparisons.pdf", height = 4, width = 8.27)

df_compare %>%
  ggplot(aes(x = NN + 1, y = JAGS + 1)) +
  ggrastr::rasterise(geom_point(alpha = 0.5, show.legend = F), dpi = 300) +
  geom_abline(intercept = 0, slope = 1,
              linetype = "dashed", colour = "red", linewidth = 0.7) +
  facet_wrap(~ metric, scales = "free") +
  labs(
    x = "Neural Network",
    y = "Bayesian"
  ) +
  scale_x_log10() +
  scale_y_log10() +
  theme(panel.background = element_blank(),
    panel.grid.major = element_line(color = "grey85", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    axis.line = element_line(color = "black"),
    strip.background = element_blank(),
    strip.text = element_text(size = 10),
    panel.spacing = unit(0.8, "lines"),
    legend.position = "none"
  )

dev.off()

# Observed vs predicted
# Observed count per site-year: mean across visits
y_obs <- apply(y, c(1, 2), mean, na.rm = TRUE)

df_obs_pred <- bind_rows(
  data.frame(
    observed = as.vector(y_obs),
    predicted = as.vector(N_nn),
    type = "Neural Network"
  ),
  data.frame(
    observed = as.vector(y_obs),
    predicted = as.vector(N_bayes),
    type = "Bayesian"
  )
)

pdf("../figures/obs_vs_pred.pdf", height = 5.83, width = 8.27)

ggplot(df_obs_pred, aes(x = observed + 1, y = predicted + 1)) +
  ggrastr::rasterise(geom_point(alpha = 0.5), dpi = 300) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "red") +
  facet_wrap(~ type) +
  labs(
    x = "Observed count",
    y = "Predicted abundance"
  ) +
  scale_x_log10() +
  scale_y_log10() +
  theme(
    panel.background = element_blank(),
    panel.grid.major = element_line(color = "grey85", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    axis.line = element_line(color = "black"),
    strip.background = element_blank(),
    strip.text = element_text(size = 10)
  )

dev.off()



