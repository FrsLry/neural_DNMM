library(reticulate)
# use_condaenv("C:/ProgramData/anaconda3", required = TRUE)
library(jagsUI)
library(rjags)
np <- import("numpy")

# Write model
cat(file = "../JAGS/DM.txt","
model {

# Priors
# alpha.lam <- log(mean.lambda)
# mean.lambda ~ dunif(0, 100) # Initial site-specific abundance
alpha.lam ~ dnorm(0,0.1)
alpha.phi ~ dnorm(0,0.1) # Apparent survival (or omega)
alpha.gamma ~ dnorm(0,0.1) # Per-capita recruitment rate
alpha.p ~ dnorm(0,0.1) # Detection probability
beta.lam ~ dnorm(0,0.1) # Coefs of 5 covariates
beta.lam.square ~ dnorm(0,0.1)
beta.phi ~ dnorm(0,0.1)
beta.phi.square ~ dnorm(0,0.1)
beta.gamma ~ dnorm(0,0.1)
beta.gamma.square ~ dnorm(0,0.1)
beta.p ~ dnorm(0,0.1)
beta.p.square ~ dnorm(0,0.1)

# Likelihood
for(i in 1:nsites){

  # State process: initial condition
  N[i,1] ~ dpois(lambda[i])

  log(lambda[i]) <- alpha.lam + beta.lam * x[i] + beta.lam.square * x[i] * x[i]
  logit(phi[i]) <- alpha.phi + beta.phi*x[i] + beta.phi.square * x[i] * x[i]
  log(gamma[i]) <- alpha.gamma + beta.gamma*x[i] + beta.gamma.square * x[i] * x[i]

  # State process: transition model
  for(t in 1:(nyears-1)){
    S[i,t+1] ~ dbin(phi[i], N[i,t])
    R[i,t+1] ~ dpois(gamma[i])
    N[i,t+1] <- S[i,t+1] + R[i,t+1]
  }

# Observation process
 for(t in 1:nyears){
  for(j in 1:nsurveys){
  logit(p[i,t,j]) <- alpha.p + beta.p*x[i] + beta.p.square*x[i]* x[i]
  C[i,t,j] ~ dbin(p[i,t,j], N[i,t])
  }
 }
}

# Derived quantities
mean.phi <- ilogit(alpha.phi)
mean.gamma <- exp(alpha.gamma)
mean.p <- ilogit(alpha.p)
}
")

dirs <- list.dirs("../simulated_data/")[-1]

for(dir in dirs){

  if(!file.exists(paste0(dir, "/jagsOut.rds"))){

    y <- np$load(paste0(dir, "/y.npy"))
    x <- as.vector(np$load(paste0(dir, "/x.npy")))
    n <- np$load(paste0(dir, "/n.npy"))
    s <- np$load(paste0(dir, "/s.npy"))
    r <- np$load(paste0(dir, "/r.npy"))

    # Initial values that usually seem to work
    R1 <- apply(y, c(1,2), max) + 10 # Use observed max. counts + 10
    # as inits for recruitment
    R1[,1] <- NA
    Nst <- apply(y, c(1,2), max) + 2
    Nst[,2:ncol(Nst)] <- NA
    inits <- function(){ list(N = Nst, R = R1, beta.lam = 0, beta.phi = 0,
                              beta.gamma = 0, beta.p = 0) }

    str(data <- list(C = y, nsites = dim(y)[1], nsurveys = dim(y)[3],
                     nyears = dim(y)[2], x = x))

    # Parameters monitored
    # could also monitor the latent variables: "N", "R", "S"
    params <- c("mean.lambda", "mean.gamma", "mean.phi", "mean.p",
                "beta.lam", "beta.gamma", "beta.phi", "beta.p", "beta.p.square", "alpha.lam", "alpha.phi",
                "alpha.gamma", "alpha.p", "beta.lam.square", "beta.phi.square", "beta.gamma.square"
                , "S", "R", "N")


    # MCMC settings
    # na <- 1000 ; ni <- 20000 ; nt <- 10 ; nb <- 5000 ; nc <- 3
    na <- 1000 ; ni <- 100000 ; nt <- 100 ; nb <- 75000 ; nc <- 3

    out <- jags(data, inits, params, "../JAGS/DM.txt", n.adapt = na, n.chains = nc,
                n.thin = nt, n.iter = ni, n.burnin = nb, parallel = TRUE)

    saveRDS(out, paste0(dir, "/jagsOut.rds"))

  }else{print("next")}

}

