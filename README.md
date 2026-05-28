# Neural Dynamic N-mixture Model: A deep learning framework to infer demographic rates from count data

## Description

* **Loss function and estimates of NSR:** This repository contains the scripts to fit the Dynamic N-mixture model (Dail & Madsen, 2011) in a neural network framework, following Joseph (2020). Most importantly, the **loss function** can be computed using the function `backward_likelihood()`, and **estimates** of abundance *N*, survivors *S*, and recruits *R* can be obtained with the function `estimating_NSR()`. Both functions can be found in `python/src/MLP_loss.py`. 

* **Simulations and case study:** All the simulations can be reproduced using the scripts `python/dyn_Nmixture_OSC.py` and `python/CNN.py` (for the computer vision simulations). The case study neural implementation can be found in `python/AHM_gw_cov_MLP.py`.

* **Figures:** All figures can be recreated using `R/scripts/figures.R`.

## License

Code and figures in this repository are under [CC-BY license](https://creativecommons.org/share-your-work/cclicenses/).
