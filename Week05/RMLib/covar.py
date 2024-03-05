import pandas as pd
import numpy as np
    
def Cov(df, skip_miss = True):
    if skip_miss == True:
        df = df.dropna()
    cov = df.cov()
    return cov

def Cor(df, skip_miss = True):
    if skip_miss == True:
        df = df.dropna()
    cor = df.corr()
    return cor

def CorToCov(sd, cor):
    cov =  pd.DataFrame(np.dot(np.dot(np.diag(sd), cor), np.diag(sd)), 
                        columns = cor.columns, index = cor.columns)
    return cov

def CovToCor(cov):
    var = np.diag(cov)
    var = var.astype('float64') 
    sd = np.sqrt(var)
    cor =  pd.DataFrame(np.dot(np.dot(np.diag(1 / sd), cov), np.diag(1 / sd)), 
                        columns = cov.columns, index = cov.columns)
    return sd, cor
    
def ewVar(x, lam):
    # subtract mean
    x = x - np.mean(x)
    # x is tuple
    m = len(x)
    w = np.empty(m)

    for i in range(m):
        w[i] = (1 - lam) * lam ** (m - i - 1)
    w /= np.sum(w)
    return np.dot(w, x**2)

def ewCovar(df, lam):
    # subtract mean
    df = df - np.mean(df, axis=0)
    m, n = df.shape
    w = np.empty(m)

    for i in range(m):
        w[i] = (1 - lam) * lam ** (m - i - 1)
    w /= np.sum(w)
    w = w.reshape(-1,1)
    return (w * df).T @ df

def ewCorr(df, lam):
    ew_covar = ewCovar(df, lam)
    sd, ew_corr = CovToCor(ew_covar)
    return ew_corr

def ewCovCor(df, lam_cov, lam_cor):
    ew_cov = ewCovar(df, lam_cov)
    ew_var = np.diag(ew_cov)
    ew_var = ew_var.astype('float64') 
    ew_sd = np.sqrt(ew_var)
    ew_cor = ewCorr(df, lam_cor)
    ew_cov_cor = pd.DataFrame(np.dot(np.dot(np.diag(ew_sd), ew_cor), np.diag(ew_sd)), 
                              columns = df.columns, index = df.columns)
    
    return ew_cov_cor



