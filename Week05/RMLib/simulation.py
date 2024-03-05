import pandas as pd
import numpy as np
import numpy.linalg as npla

from . import covar, non_psd

def simNormal(cov, times, method = None):
    
    np.random.seed(4)
    
    if method == "near_psd":
        cov = non_psd.nearPSDCov(cov)
    elif method == "higham_nearestPSD":
        cov = non_psd.higham_nearestPSDCov(cov)
    n = cov.shape[0]
    
    sim_df = pd.DataFrame(np.random.multivariate_normal(np.zeros(n), cov, times), 
                          columns = cov.columns)

    return sim_df

def simPca(cov, times, pctExp=1):
    
    n = cov.shape[0]
    mean = 0
    vals, vecs = npla.eig(cov)
    indices = np.argsort(vals)[::-1]  # Get the indices that sort vals in descending order.
    vals = vals[indices]  # Apply the sorting indices to vals.
    vecs = vecs[:, indices] 
    total_val = np.sum(vals)
    posv = np.where(vals >= 1e-8)[0]

    if pctExp < 1:
        nval = 0
        pct = 0
        # figure out how many factors are needed
        for i in posv:
            pct += vals[i]/total_val
            nval += 1
            if pct >= pctExp:
                break
            
        posv = posv[:nval]
        
    vals = vals[posv]
    vecs = vecs[:, posv]
    
    # Print total variance explained by selected components
    # print(f"Simulating with {len(posv)} PC Factors: {np.sum(vals) / total_val * 100}% total variance explained")
        
    B = vecs @ np.diag(np.sqrt(vals))
    r = np.random.randn(len(vals), times)
    out = (B @ r).T
    sim_df = pd.DataFrame(out)
        
    return sim_df