import pandas as pd
import numpy as np
from scipy import optimize
from scipy.stats import norm, t

from .covar import ewCovar
from .fit_model import fit_normal, fit_general_t

# helper function return VaR normal df
def VaR_norm_df(alpha, mu, sigma):
    VaR = -norm.ppf(alpha, loc = mu, scale = sigma)
    diff = -norm.ppf(alpha, loc = 0, scale = sigma)
    return pd.DataFrame({"VaR Absolute": [VaR], 
                         "VaR Diff from Mean": [diff]})
    
# helper function return VaR t df
def VaR_t_df(alpha, nu, mu, sigma):
    VaR = -t.ppf(alpha, df = nu, loc = mu, scale = sigma)
    diff = -t.ppf(alpha, df = nu, loc = 0, scale = sigma)
    return pd.DataFrame({"VaR Absolute": [VaR], 
                         "VaR Diff from Mean": [diff]})
    
# def VaR_normal_distribution(ror, alpha=0.05):
#     params = fit_normal(ror)
#     mu = params.loc[0, "mu"]
#     sigma = params.loc[0, "sigma"]
#     return VaR_norm_df(alpha, mu, sigma)

def VaR_normal_distribution(ror, alpha=0.05):
    ror = ror.to_numpy()
    mu = np.mean(ror)
    ror = ror - mu
    sigma = np.std(ror)
    return VaR_norm_df(alpha, mu, sigma)
    
def Var_normal_distribution_EW(ror, lam, alpha=0.05):
    mu = np.mean(ror)
    ror = ror - mu
    ew_sigma2 = ewCovar(ror, lam).to_numpy()[0,0]
    ew_sigma = ew_sigma2 ** 0.5
    return VaR_norm_df(alpha, mu, ew_sigma)
    
def VaR_t_distribution(ror, alpha=0.05):
    params = fit_general_t(ror)
    mu = params.loc[0, "mu"]
    sigma = params.loc[0, "sigma"]
    nu = params.loc[0, "nu"]
    return VaR_t_df(alpha, nu, mu, sigma)
    
def VaR_MLE_t(ror, alpha=0.05):
    ror = ror.to_numpy()
    mu = np.mean(ror)
    ror = ror - mu
    def ll_t(params):
        df = params[0]
        s = params[1]
        ll = np.sum(t.logpdf(ror, df = df, loc = 0, scale = s))
        return -ll
    
    s = np.std(ror)
    df = 1
    params = [df, s]
    bnds = ((0, None), (1e-9, None))
    res = optimize.minimize(ll_t, params, bounds = bnds, options={"disp": False})
    nu = res.x[0]
    sigma = res.x[1]
    return VaR_t_df(alpha, nu, mu, sigma)

# def VaR_historic(ror, alpha=0.05):
#     ror0 = ror - np.mean(ror)
#     return pd.DataFrame({"VaR Absolute": [-ror.quantile(alpha)['x']], 
#                          "VaR Diff from Mean": [-ror0.quantile(alpha)['x']]})
    
def VaR_historic(ror, alpha=0.05):
    ror = ror.to_numpy()
    indices = np.random.choice(len(ror), size = 10000, replace = True)
    sim_ror = ror[indices]
    sim_ror0 = sim_ror - np.mean(sim_ror)
    return pd.DataFrame({"VaR Absolute": [-np.percentile(sim_ror, alpha*100)], 
                         "VaR Diff from Mean": [-np.percentile(sim_ror0, alpha*100)]})
    
def VaR_simulation(ror, alpha=0.05, n = 10000):
    params = fit_general_t(ror)
    mu = params.loc[0, "mu"]
    sigma = params.loc[0, "sigma"]
    nu = params.loc[0, "nu"]
    sim = np.random.standard_t(nu, n)
    sim = sim * sigma + mu
    return VaR_t_distribution(sim, alpha)
