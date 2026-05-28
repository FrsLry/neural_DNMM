import torch
import numpy as np
from torch.distributions import Binomial, Poisson

def transition_matrix(n_max, phi_hat, gamma_hat, nt, nsite, recruitment="absolute"):
    """
    Compute the transition probability matrix P(n, m) for the dynamic N-mixture model.

    Each element P[n, m] represents the probability of transitioning from n individuals 
    at time t to m individuals at time t+1 at a given site. The transition is composed 
    of two processes:
        1) Survival of individuals with probability phi 
        2) Recruitment of new individuals with rate gamma

    Instead of using PyTorch distributions (Binomial/Poisson objects), this function computes 
    log-probabilities manually for computational efficiency.

    Args:
        n_max (int): Maximum possible abundance of individuals.
        phi_hat (torch.Tensor): [nsite, nt-1] survival probabilities for each site and time step.
        gamma_hat (torch.Tensor): [nsite, nt-1] recruitment rates for each site and time step.
        nt (int): Total number of time steps.
        nsite (int): Number of spatial sites.
        
    Returns:
        P (torch.Tensor): Transition matrices of shape [nsite, nt-1, n_max, n_max],
                          where P[i, t, n, m] is the probability of going from n to m individuals 
                          at site i between time t and t+1. This is then used in the backward likelihood computation to marginalize over latent states.
    """
    # Define possible values of n, m, s (latent counts and possible survivors)
    n_vals = torch.arange(n_max, device=phi_hat.device) # possible n at time t
    m_vals = torch.arange(n_max, device=phi_hat.device) # possible m at time t+1
    s_vals = torch.arange(n_max, device=phi_hat.device) # possible s (survivors)

    # Create a meshgrid over all combinations of n, m, s:
    # Resulting shape: [n_max, n_max, n_max], where axes 0=n, 1=m, 2=s
    N, M, S = torch.meshgrid(n_vals, m_vals, s_vals)

    # Expand to [1, n_max, n_max, n_max] to allow site broadcasting
    N = N.unsqueeze(0)
    M = M.unsqueeze(0)
    S = S.unsqueeze(0)

    # Create a mask for invalid (n, m, s) combinations:
    # S cannot be greater than n (cannot survive more than initial)
    # S cannot be greater than m (cannot contribute more survivors than next abundance)
    invalid_mask = (S > N) | (S > M)  # shape: [1, n_max, n_max, n_max]

    # Initialize the transition matrix tensor for all sites and time steps
    P = torch.zeros(nsite, nt - 1, n_max, n_max, device=phi_hat.device)

    # Loop over time steps
    for t in range(nt - 1):

        # Get phi and gamma for all sites at time t
        phi = phi_hat[:, t].view(nsite, 1, 1, 1)  # [nsite,1,1,1]
        gamma_base = gamma_hat[:, t].view(nsite, 1, 1, 1)  # [nsite,1,1,1]

        # Expand N and S across the site dimension to shape [nsite, n_max, n_max, n_max]
        N_b = N.expand(nsite, -1, -1, -1)
        S_b = S.expand(nsite, -1, -1, -1)

        if recruitment == "absolute":
            gamma = gamma_base
        elif recruitment == "per_capita":
            gamma = gamma_base * N_b
        else:
            raise ValueError("recruitment must be 'absolute' or 'per_capita'")

        # Compute the binomial coefficient (in log space) using lgamma:
        # The binomial coefficient C(N, S) = N! / [S! * (N - S)!] represents the number of ways
        # to choose S survivors from N individuals.
        # To avoid computing factorials directly (which can overflow for large N),
        # we use the log-gamma function: lgamma(k + 1) = log(k!)
        # (Note: lgamma(k) returns log((k-1)!), so we use k+1 to get log(k!))
        #
        # Thus:
        # log(C(N, S)) = log(N!) - log(S!) - log((N - S)!)
        #               = lgamma(N + 1) - lgamma(S + 1) - lgamma(N - S + 1)
        log_binom_coeff = (torch.lgamma(N_b + 1) - torch.lgamma(S_b + 1) - 
                           torch.lgamma(N_b - S_b + 1))
        
        # Compute log P(S survivors | N initial, phi survival prob):
        # Binomial log PMF:
        # log P(S | N, phi) = log_binom_coeff + S * log(phi) + (N - S) * log(1 - phi)
        log_p_survive = log_binom_coeff + S_b * torch.log(phi) + (N_b - S_b) * torch.log1p(-phi)

        # Set log P to -inf for invalid (n, m, s) combinations:
        log_p_survive[invalid_mask.expand(nsite, -1, -1, -1)] = -float('inf')

        # Compute recruits:
        # Number of recruits = m - s (total next abundance - survivors)
        R = M - S
        R_clamped = torch.clamp(R, min=0) # can't have negative recruits
        R_b = R_clamped.expand(nsite, -1, -1, -1)

        # Compute log P(R recruits | gamma rate):
        # Poisson log PMF:
        # log P(R | gamma) = R * log(gamma) - gamma - lgamma(R + 1)
        log_p_recruit = (R_b * torch.log(gamma + 1e-20) - gamma - torch.lgamma(R_b + 1))

        # Total log-probability of survival + recruitment for each (n, m, s):
        log_p = log_p_survive + log_p_recruit

        # Sum over all possible s (number of survivors) to get P(n->m):
        P_log = torch.logsumexp(log_p, dim=3)  # [nsite, n_max, n_max]

        # Exponentiate to get P (back to probability space)
        P[:, t, :, :] = torch.exp(P_log)

        # For large n_max:
        torch.cuda.empty_cache()

    return P

def backward_likelihood(y, lambda_hat, p_hat, phi_hat, gamma_hat, n_max, nt, recruitment="absolute"):
    """
    Compute the log-likelihood of observed data y under a dynamic N-mixture model
    using the 'backward' algorithm
    
    Args:
        y (torch.Tensor): [R, T, J] tensor of observed counts
                          R = number of sites
                          T = number of primary time steps
                          J = number of repeated counts per primary time step (detection repeats)
                          
        lambda_hat (torch.Tensor): [R] Poisson rate parameters for initial abundance N_{i1} per site.
        p_hat (torch.Tensor): [R, T] detection probabilities per site and time.
        phi_hat (torch.Tensor): [R, T-1] survival probabilities per site and time.
        gamma_hat (torch.Tensor): [R, T-1] recruitment rates per site and time.
        n_max (int): Maximum value for latent abundance N (truncation of the state space).

    Returns:
        log_likelihood (torch.Tensor): [R] tensor, log-likelihood value per site.
    """

    R, T, J = y.shape # Number of sites, time steps, and repeats

    # Define all possible abundance states N = 0, 1, ..., n_max-1
    nval = torch.arange(n_max, device=y.device).view(1, 1, -1, 1)  # [1, 1, n_max, 1]

    # Expand y and p for broadcasting
    missing = ~torch.isfinite(y)                                # True where y is NaN
    y_filled = torch.where(missing, torch.zeros_like(y), y)     # dummy counts for log_prob
    y_obs = y_filled.unsqueeze(2)  # [R, T, 1, J] 
    
    # detection probabilities: either same or different values of covariates per survey 
    if p_hat.ndim == 2:
        # one p per site-year
        p = p_hat.unsqueeze(2).unsqueeze(-1)   # [R, T, 1, 1]
    elif p_hat.ndim == 3:
        # one p per site-year-survey
        if p_hat.shape[2] != J:
            raise ValueError(
                f"p_hat has shape {p_hat.shape}, but y has {J} surveys"
            )
        p = p_hat.unsqueeze(2)                 # [R, T, 1, J]
    else:
        raise ValueError(
            f"p_hat must have shape [R,T] or [R,T,J], got {p_hat.shape}"
        )

    # Compute detection probabilities
    log_binom = Binomial(total_count=nval, probs=p, validate_args=False).log_prob(y_obs)  # [R, T, n_max, J]
    log_binom = torch.where(missing.unsqueeze(2), torch.zeros_like(log_binom), log_binom)

    # g1: detection proba. Sum over J repeats
    g1_log = log_binom.sum(dim=3)  # [R, T, n_max]
   
    # g2: initial abundance
    g2_log = Poisson(lambda_hat).log_prob(nval.view(1, -1))

    # Compute transition matrix P(n, m) for each time step (bottleneck: computationally and memory expensive)
    P = transition_matrix(n_max, phi_hat, gamma_hat, nt, R, recruitment=recruitment)
    # For stability, log(P + epsilon)
    P = torch.log(P + 1e-20)
    
    # Initialize g* for the backward recursion. g*[t, n] will store the log-probability of observing future data given N_t=n
    g_star = torch.zeros(R, n_max, device = y.device) 

    # Recursive computation of g*
    for t in reversed(range(1, T)):
        g1_t = g1_log[:, t, :]  # [R, n_max], detection likelihood at time t
        P_t = P[:, t - 1, :, :]  # [R, n_max, n_max], transition from t-1 to t

        # tmp: combine detection at t and future contribution (g_star)
        tmp = g1_t + g_star  # [R, n_max]

        # summands: sum over all possible N_t values
        # P_t: [R, n_max (n), n_max (m)] + tmp.unsqueeze(1): [R, 1, n_max]
        # This results in: [R, n_max (n), n_max (m)] ready to sum over m (next states)
        summands = P_t +  tmp.unsqueeze(1)

        # Marginalize over N_{t+1} (m):
        # result: [R, n_max], total probability of reaching all possible states N_t
        g_star = torch.logsumexp(summands, dim=2)

    # Final marginalization over N_1 (initial latent abundance):
    # Incorporate detection at time 1 (g1_log[:,0,:]), Poisson prior (g2_log), and g_star
    final = g1_log[:, 0, :] + g2_log + g_star  # [R, n_max]

    # Sum over N_1 to get total likelihood per site
    log_likelihood = torch.logsumexp(final, dim=1)

    return log_likelihood

def estimating_NSR(y, lambda_hat, p_hat, phi_hat, gamma_hat, batch_size=None, recruitment="absolute"):
    """
    Estimating expected abundance N, survivors S, and recruits R at each time step using a forward-backward algorithm

    Args:
        y : Tensor [R, T, J]
        lambda_hat : Tensor [R] or [R,1]
        p_hat : Tensor [R,T] or [R,T,J]
        phi_hat : Tensor [R,T-1]
        gamma_hat : Tensor [R,T-1]
        batch_size : int or None
            If not None, split computation over the site dimension.
        
    Returns:
        E_N : Tensor [R,T]
        E_S : Tensor [R,T-1]
        E_R : Tensor [R,T-1]    
    """

    R, T, J = y.shape
    device = y.device
    n_max = np.nanmax(y.cpu().numpy()).astype(np.int32) + 60

    # Batch wrapper ######
    if batch_size is not None and R > batch_size:
        E_N_list, E_S_list, E_R_list = [], [], []

        for start in range(0, R, batch_size):
            end = min(start + batch_size, R)

            out = estimating_NSR(
                y[start:end],
                lambda_hat[start:end],
                p_hat[start:end],
                phi_hat[start:end],
                gamma_hat[start:end],
                batch_size=None,
                recruitment=recruitment,
            )

            E_N_b, E_S_b, E_R_b= out
            E_N_list.append(E_N_b)
            E_S_list.append(E_S_b)
            E_R_list.append(E_R_b)

            if device.type == "cuda":
                torch.cuda.empty_cache()

        return (
            torch.cat(E_N_list, dim=0),
            torch.cat(E_S_list, dim=0),
            torch.cat(E_R_list, dim=0),
        )

    nt = T

    if lambda_hat.ndim == 2:
        lambda_hat = lambda_hat.squeeze(1)

    nvals = torch.arange(n_max, device=device, dtype=torch.float32)  # [n_max]

    # Emission likelihoods:
    # Compute log p(y_rt | N_t = n) for all sites, times and latent states
    # Handles missing surveys by zeroing their contribution
    missing = ~torch.isfinite(y)
    y_filled = torch.where(missing, torch.zeros_like(y), y)

    nval4 = torch.arange(n_max, device=device).view(1, 1, -1, 1)  # [1,1,n,1]
    y_obs = y_filled.unsqueeze(2)                                  # [R,T,1,J]

    if p_hat.ndim == 2:
        p = p_hat.unsqueeze(2).unsqueeze(-1)                       # [R,T,1,1]
    elif p_hat.ndim == 3:
        if p_hat.shape[2] != J:
            raise ValueError(f"p_hat has shape {p_hat.shape}, but y has {J} surveys")
        p = p_hat.unsqueeze(2)                                     # [R,T,1,J]
    else:
        raise ValueError(f"p_hat must have shape [R,T] or [R,T,J], got {p_hat.shape}")

    log_binom = Binomial(total_count=nval4, probs=p, validate_args=False).log_prob(y_obs)
    log_binom = torch.where(missing.unsqueeze(2), torch.zeros_like(log_binom), log_binom)
    g1_log = log_binom.sum(dim=3)                                  # [R,T,n_max]

    # Prior and transition matrices:
    # Build the Poisson prior over N_1 and construct transition matrices P(n->m)
    log_prior = Poisson(lambda_hat.unsqueeze(1)).log_prob(nvals.view(1, -1))  # [R, n_max]

    P = transition_matrix(n_max, phi_hat, gamma_hat, nt, R, recruitment=recruitment)                    # [R,T-1,n,m]
    logP = torch.log(P + 1e-20)

    # Forward pass (alpha):
    # Compute filtered log-probabilities log alpha_t(n) = log p(N_t=n | y_1:t)
    log_alpha = torch.zeros(R, T, n_max, device=device)

    log_alpha[:, 0, :] = log_prior + g1_log[:, 0, :]
    log_alpha[:, 0, :] -= torch.logsumexp(log_alpha[:, 0, :], dim=1, keepdim=True)

    for t in range(1, T):
        log_alpha_pred = torch.logsumexp(
            log_alpha[:, t - 1, :].unsqueeze(2) + logP[:, t - 1, :, :],
            dim=1
        )

        log_alpha[:, t, :] = log_alpha_pred + g1_log[:, t, :]
        log_alpha[:, t, :] -= torch.logsumexp(log_alpha[:, t, :], dim=1, keepdim=True)

    # Backward pass (beta):
    # Compute log-beta, the contribution of future observations to each state
    log_beta = torch.zeros(R, T, n_max, device=device)
    log_beta[:, T - 1, :] = 0.0

    for t in range(T - 2, -1, -1):
        tmp = (
            logP[:, t, :, :] +
            g1_log[:, t + 1, :].unsqueeze(1) +
            log_beta[:, t + 1, :].unsqueeze(1)
        )

        log_beta[:, t, :] = torch.logsumexp(tmp, dim=2)
        log_beta[:, t, :] -= torch.logsumexp(log_beta[:, t, :], dim=1, keepdim=True)

    # Smoothed state posterior:
    # Combine forward and backward terms to obtain p(N_t=n | y_1:T),
    # then compute the posterior mean E[N_t].
    log_gamma_state = log_alpha + log_beta
    log_gamma_state -= torch.logsumexp(log_gamma_state, dim=2, keepdim=True)
    P_N_smooth = torch.exp(log_gamma_state)

    E_N = torch.sum(P_N_smooth * nvals.view(1, 1, -1), dim=2)

    # Smoothed pairwise transitions:
    # Compute XI_t(n,m) = p(N_t=n, N_{t+1}=m | y_1:T) for expectations over transitions.
    XI = torch.zeros(R, T - 1, n_max, n_max, device=device)

    for t in range(T - 1):
        log_xi = (
            log_alpha[:, t, :].unsqueeze(2) +
            logP[:, t, :, :] +
            g1_log[:, t + 1, :].unsqueeze(1) +
            log_beta[:, t + 1, :].unsqueeze(1)
        )

        log_xi -= torch.logsumexp(log_xi.view(R, -1), dim=1, keepdim=True).view(R, 1, 1)
        XI[:, t, :, :] = torch.exp(log_xi)

    # Expected survivors and recruits:
    # For each interval compute the conditional distribution of survivors S given (n,m),
    # then take expectations under XI to get E[S] and E[R]
    E_S = torch.zeros(R, T - 1, device=device)
    E_R = torch.zeros(R, T - 1, device=device)

    n_grid = torch.arange(n_max, device=device)
    m_grid = torch.arange(n_max, device=device)
    s_grid = torch.arange(n_max, device=device)

    N, M, S = torch.meshgrid(n_grid, m_grid, s_grid, indexing='ij')
    N = N.unsqueeze(0)
    M = M.unsqueeze(0)
    S = S.unsqueeze(0)

    invalid_mask = (S > N) | (S > M)

    for t in range(T - 1):
        phi = phi_hat[:, t].view(R, 1, 1, 1)
        gamma_base = gamma_hat[:, t].view(R, 1, 1, 1) 
        
        N_b = N.expand(R, -1, -1, -1)
        S_b = S.expand(R, -1, -1, -1)

        if recruitment == "absolute":
            gamma = gamma_base
        elif recruitment == "per_capita":
            gamma = gamma_base * N_b
        else:
            raise ValueError("recruitment must be 'absolute' or 'per_capita'")

        log_binom_coeff = (
            torch.lgamma(N_b + 1) -
            torch.lgamma(S_b + 1) -
            torch.lgamma(N_b - S_b + 1)
        )

        log_p_survive = (
            log_binom_coeff +
            S_b * torch.log(phi) +
            (N_b - S_b) * torch.log1p(-phi)
        )
        log_p_survive[invalid_mask.expand(R, -1, -1, -1)] = -float("inf")

        Rcount = M - S
        Rcount_clamped = torch.clamp(Rcount, min=0)
        R_b = Rcount_clamped.expand(R, -1, -1, -1)

        log_p_recruit = (
            R_b * torch.log(gamma + 1e-20) -
            gamma -
            torch.lgamma(R_b + 1)
        )

        log_joint_s = log_p_survive + log_p_recruit
        log_norm_nm = torch.logsumexp(log_joint_s, dim=3, keepdim=True)
        log_cond_s_given_nm = log_joint_s - log_norm_nm

        P_s_given_nm = torch.exp(log_cond_s_given_nm)

        E_S_given_nm = torch.sum(
            P_s_given_nm * s_grid.view(1, 1, 1, -1),
            dim=3
        )

        E_R_given_nm = m_grid.view(1, 1, -1) - E_S_given_nm

        E_S[:, t] = torch.sum(XI[:, t, :, :] * E_S_given_nm, dim=(1, 2))
        E_R[:, t] = torch.sum(XI[:, t, :, :] * E_R_given_nm, dim=(1, 2))

    return E_N, E_S, E_R
