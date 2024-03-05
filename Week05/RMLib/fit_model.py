import numpy as np
import pandas as pd
from scipy import stats
from scipy import optimize
import statsmodels.api as sm

def fit_normal(data):
    mu, sigma = stats.norm.fit(data)
    return pd.DataFrame({
        "mu": [mu],
        "sigma": [sigma]
    })

def fit_general_t(data):
    df, loc, scale = stats.t.fit(data)
    return pd.DataFrame({
        "mu": [loc],
        "sigma": [scale],
        "nu": [df]
    })
    
def MLE_t(X, Y):
    X = sm.add_constant(X)
    def ll_t(params):
        df = params[0]
        s = params[1]
        b = params[2:]
        e = Y - np.dot(X, b)
        ll = np.sum(stats.t.logpdf(e, df=df, loc=0, scale=s))
        return -ll
    beta = np.zeros(X.shape[1])
    s = np.std(Y - np.dot(X, beta))
    df = 1
    params = [df, s]
    for i in beta:
        params.append(i)
    bnds = ((0, None), (1e-9, None), (None, None), (None, None), (None, None), (None, None))
    res = optimize.minimize(ll_t, params, bounds=bnds, options={"disp": False})
    beta_mle = res.x[2:]
    return beta_mle

def fit_regression_t(data):
    X = data.iloc[:,:-1]
    Y = data.iloc[:,-1]
    betas = MLE_t(X, Y)
    X = sm.add_constant(X)
    e = Y - np.dot(X, betas)
    df, loc, scale = stats.t.fit(e)
    out = {
        "mu": [loc],
        "sigma": [scale],
        "nu": [df]
    }
    for i in range(len(betas)):
        if i == 0:
            out["Alpha"] = betas[i]
        else:
            out["B" + str(i)] = betas[i]
    return pd.DataFrame(out)


