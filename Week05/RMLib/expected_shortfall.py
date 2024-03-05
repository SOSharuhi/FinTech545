import pandas as pd
import numpy as np
from scipy import optimize
from scipy.stats import norm, t

from .covar import ewCovar
from .fit_model import fit_normal, fit_general_t
from . import value_at_risk as var

# helper function ES normal df
def ES_norm_df(VaR, mu, sigma):
    x_a = -VaR.loc[0, "VaR Absolute"]
    x_d = -VaR.loc[0, "VaR Diff from Mean"]
    ES = -norm.expect(ub = x_a, loc = mu, scale = sigma, conditional = True)
    diff = -norm.expect(ub = x_d, loc = 0, scale = sigma, conditional = True)
    return pd.DataFrame({"ES Absolute": [ES], 
                         "ES Diff from Mean": [diff]})

# helper function ES t df
def ES_t_df(VaR, nu, mu, sigma):
    x_a = -VaR.loc[0, "VaR Absolute"]
    x_d = -VaR.loc[0, "VaR Diff from Mean"]
    ES = -t.expect(ub = x_a, args=(nu,), loc = mu, scale = sigma, conditional = True)
    diff = -t.expect(ub = x_d, args=(nu,), loc = 0, scale = sigma, conditional = True)
    return pd.DataFrame({"ES Absolute": [ES], 
                         "ES Diff from Mean": [diff]})

# def ES_normal_distribution(ror, alpha=0.05):
#     params = fit_normal(ror)
#     mu = params.loc[0, "mu"]
#     sigma = params.loc[0, "sigma"]
#     VaR = var.VaR_normal_distribution(ror, alpha)
#     return ES_norm_df(VaR, mu, sigma)

def ES_normal_distribution(ror, alpha=0.05):
    ror_np = ror.to_numpy()
    mu = np.mean(ror_np)
    ror_np = ror_np - mu
    sigma = np.std(ror_np)
    VaR = var.VaR_normal_distribution(ror, alpha)
    return ES_norm_df(VaR, mu, sigma)

def ES_t_distribution(ror, alpha=0.05):
    params = fit_general_t(ror)
    mu = params.loc[0, "mu"]
    sigma = params.loc[0, "sigma"]
    nu = params.loc[0, "nu"]
    VaR = var.VaR_t_distribution(ror, alpha)
    return ES_t_df(VaR, nu, mu, sigma)

def ES_normal_distribution_EW(ror, lam, alpha=0.05):
    mu = np.mean(ror)
    ror = ror - mu
    ew_sigma2 = ewCovar(ror, lam).to_numpy()[0,0]
    ew_sigma = ew_sigma2 ** 0.5
    VaR = var.Var_normal_distribution_EW(ror, lam, alpha)
    return ES_norm_df(VaR, mu, ew_sigma)

def ES_historic(ror, alpha=0.05):
    ror = ror.to_numpy()
    indices = np.random.choice(len(ror), size = 10000, replace = True)
    sim_ror = ror[indices]
    sim_ror0 = sim_ror - np.mean(sim_ror)
    x_a = np.percentile(sim_ror, alpha*100)
    x_d = np.percentile(sim_ror0, alpha*100)
    
    ES = -np.mean(sim_ror[sim_ror <= x_a])
    diff = -np.mean(sim_ror0[sim_ror0 <= x_d])
    return pd.DataFrame({"ES Absolute": [ES], 
                         "ES Diff from Mean": [diff]})
    
def ES_simulation(ror, alpha=0.05, n = 10000):
    params = fit_general_t(ror)
    mu = params.loc[0, "mu"]
    sigma = params.loc[0, "sigma"]
    nu = params.loc[0, "nu"]
    VaR = var.VaR_simulation(ror, alpha, n)
    return ES_t_df(VaR, nu, mu, sigma)


