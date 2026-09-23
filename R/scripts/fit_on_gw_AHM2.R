### Fitting the model on the Swiss Green Woodpecker data from Kery and Royle Vol. 2 (2020)
library(jagsUI)
library(AHMbook)

source(file = "AHM_data/AHM2_02.02.R")

# Load shared train / validation / test split
split_df <- readRDS("AHM_data/gw_site_split_seed347.rds")

train_idx <- split_df$site_R[split_df$split == "train"]
val_idx   <- split_df$site_R[split_df$split == "val"]
test_idx  <- split_df$site_R[split_df$split == "test"]

# Keep only training sites for Bayesian model fitting
C <- C[train_idx, , , drop = FALSE]
DATE <- DATE[train_idx, , , drop = FALSE]
INT <- INT[train_idx, , , drop = FALSE]
elev <- elev[train_idx]
forest <- forest[train_idx]
route_length <- peckers$route.length[train_idx]

cat("Train sites:", length(train_idx), "\n")
cat("Validation sites:", length(val_idx), "\n")
cat("Test sites:", length(test_idx), "\n")

# Bundle data
str(bdata <- list(
  C = C,
  nsites = dim(C)[1],
  nsurveys = dim(C)[2],
  nyears = dim(C)[3],
  elev = as.vector(elev),
  forest = as.vector(forest),
  DATE = DATE,
  length = route_length,
  INT = INT
))

# Specify model in JAGS language
cat(file = "../JAGS/DM_gw.txt","
model {
  # Priors
  alpha.lam ~ dnorm(0,0.1) # Abundance parameters
  beta.elev ~ dnorm(0,0.1)
  beta.elev2 ~ dnorm(0,0.1)
  beta.for ~ dnorm(0,0.1)
  phi ~ dunif(0, 1) # App. survival (omega in paper/unmarked)
  beta.gamma ~ dnorm(0,0.1) # Recruitment model parameter
  gamma0 <- exp(beta.gamma)
  alpha.p ~ dnorm(0,0.1) # Detection probability parameters
  beta.jul ~ dnorm(0,0.1)
  beta.jul2 ~ dnorm(0,0.1)
  beta.int ~ dnorm(0,0.1)
  beta.int2 ~ dnorm(0,0.1)

  # Likelihood
  for(i in 1:nsites){
    # State process: initial condition
    log(lambda[i]) <- alpha.lam + beta.elev*elev[i] +
    beta.elev2*elev[i]*elev[i] + beta.for*forest[i]
    N[i,1] ~ dpois(lambda[i])
    # State process: transition model
    for(t in 1:(nyears-1)){
      S[i,t+1] ~ dbin(phi, N[i,t])
      R[i,t+1] ~ dpois(gamma0)             # constant recruitment
      ###R[i,t+1] ~ dpois(N[i,t] * gamma0) # per-capita recruitment
      N[i,t+1] <- S[i,t+1] + R[i,t+1]
    }

    # Observation process
    for(t in 1:nyears){
      for(j in 1:nsurveys){
        logit(p[i,j,t]) <- alpha.p + beta.jul*DATE[i,j,t] +
            beta.jul2*DATE[i,j,t]*DATE[i,j,t] + beta.int*INT[i,j,t] +
            beta.int2*INT[i,j,t]*INT[i,j,t]
        C[i,j,t] ~ dbin(p[i,j,t], N[i,t])
      }
    }
  }
  # Derived quantities
  Nbar <- mean(N)
  lambar <- mean(lambda)
}
")


# Initial values, sometimes very challenging
Rst <- apply(C, c(1,3), max, na.rm = TRUE)
Rst[Rst == '-Inf'] <- 1
Rst[,1] <- NA
Nst <- array(NA, dim = dim(Rst))
tmp <- apply(C, 1, max, na.rm = TRUE)
tmp[tmp == '-Inf'] <- 2
Nst[,1] <- tmp

# Initial values
inits <- function(){list(R = Rst, N = Nst+1, alpha.lam = rnorm(1, -1,1),
                         beta.elev = rnorm(1), beta.elev2 = rnorm(1), beta.for = rnorm(1), beta.ilen = runif(1,-1,0),
                         phi = 0.5, alpha.gamma = rnorm(1), beta.gamma = rnorm(1), sigma.gamma = 0.5, alpha.p = rnorm(1),
                         beta.jul = rnorm(1), beta.jul2 = rnorm(1), beta.int = rnorm(1), beta.int2 = rnorm(1))}

# Parameters monitored
params <-c("alpha.lam", "beta.elev", "beta.elev2", "beta.for",
           "phi", "alpha.gamma", "beta.gamma", "sigma.gamma", "gamma0",
           "alpha.p", "beta.jul", "beta.jul2", "beta.int", "beta.int2", "S", "R", "N")

# MCMC settings
na <- 1000 ; ni <- 70000 ; nt <- 4 ; nb <- 10000 ; nc <- 3
# na <- 1000 ; ni <- 7000 ; nt <- 1 ; nb <- 1000 ; nc <- 3  # ~~~ for testing, 7 mins

# try different seeds in hopes of getting one that runs
set.seed(333)

t_start <- Sys.time()

# Call JAGS (ART 83 min), check convergence and summarize posteriors
out <- jags(bdata, inits, params, "../JAGS/DM_gw.txt", n.adapt = na, n.chains = nc,
             n.thin = nt, n.iter = ni, n.burnin = nb, parallel = TRUE)

t_end <- Sys.time()

print(t_end - t_start)

# saveRDS(out, "AHM_data/jagsOut_gw_covariates_noIntens_seed347.rds")

# par(mfrow = c(2,3))  #  ~~~ replace with 'layout' argument
# traceplot(out, layout=c(2,3))
# print(out, digits=2)
