### Create train / validation / test split for the Swiss Green Woodpecker data
### This script should be run once, before fitting the neural and Bayesian models.

set.seed(42)

### Load Swiss Green Woodpecker data
source(file = "AHM_data/AHM2_02.02.R")

### Number of sites
n_sites <- dim(C)[1]

### Split proportions
train_frac <- 0.60
val_frac   <- 0.20
test_frac  <- 0.20

stopifnot(abs(train_frac + val_frac + test_frac - 1) < 1e-8)

### Create site-level split
site_R <- seq_len(n_sites)

n_test <- floor(test_frac * n_sites)
n_val  <- floor(val_frac * n_sites)

test_idx <- sort(sample(site_R, size = n_test, replace = FALSE))

remaining_idx <- setdiff(site_R, test_idx)

val_idx <- sort(sample(remaining_idx, size = n_val, replace = FALSE))

train_idx <- sort(setdiff(remaining_idx, val_idx))

### Save both R-style and Python-style indices
### R indices are 1-based; Python indices are 0-based.
split_df <- data.frame(
  site_R = site_R,
  site_python = site_R - 1L,
  split = NA_character_
)

split_df$split[split_df$site_R %in% train_idx] <- "train"
split_df$split[split_df$site_R %in% val_idx]   <- "val"
split_df$split[split_df$site_R %in% test_idx]  <- "test"

### Sanity checks
stopifnot(!any(is.na(split_df$split)))
stopifnot(sum(split_df$split == "train") +
            sum(split_df$split == "val") +
            sum(split_df$split == "test") == n_sites)

print(table(split_df$split))

### Save split
saveRDS(split_df, file = "AHM_data/gw_site_split_seed42.rds")
