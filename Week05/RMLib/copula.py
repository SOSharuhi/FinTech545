import pandas as pd
import numpy as np
from scipy import stats

import RMLib.returns as returns
import RMLib.fit_model as fit
import RMLib.simulation as sim

def aggRisk(simRet, values, models, alpha=0.05):
    risk_df = pd.DataFrame(columns = ['Stock', 'VaR95', 'ES95', 'VaR95_Pct', 'ES95_Pct'])
    stocks = values['Stock'].unique()
    m = stocks.shape[0]

    total_weight = np.sum(values['CurrentValue'])
    total_value = 0
    weight = np.zeros(m)
    # each stock
    for i in range(m):
        stock = stocks[i]
        risk_df.loc[i, 'Stock'] = stock
        value = values[values['Stock'] == stock]
        weight[i] = weight[i] + sum(value['CurrentValue'])
        stock_cv = value['CurrentValue'].unique()
        total_value += stock_cv
        dist = value['Dist'].unique()
        pnl = value['pnl']
    
        # VaR
        risk_df.loc[i, 'VaR95'] = -np.percentile(pnl, alpha * 100)
        risk_df.loc[i, 'VaR95_Pct'] = risk_df.loc[i, 'VaR95'] / stock_cv
        
        # ES
        if (dist == 'normal'):
            risk_df.loc[i, 'ES95_Pct'] = -stats.norm.expect(ub = -risk_df.loc[i, 'VaR95_Pct'], 
                                                            loc = models[stock].loc[0, "mu"],
                                                            scale = models[stock].loc[0, "sigma"],
                                                            conditional=True)
        elif (dist == 't'):
            risk_df.loc[i, 'ES95_Pct'] = -stats.t.expect(ub = -risk_df.loc[i, 'VaR95_Pct'], 
                                                        args = (models[stock].loc[0, "nu"],),
                                                        loc = models[stock].loc[0, "mu"],
                                                        scale = models[stock].loc[0, "sigma"],
                                                        conditional=True)
            
        risk_df.loc[i, 'ES95'] = risk_df.loc[i, 'ES95_Pct'] * stock_cv

    # total portfolio
    weight = weight / total_weight
    risk_df.loc[m, 'Stock'] = 'Total'
    total_mu = np.mean(np.sum(weight * simRet, axis=1))
    total_sigma = np.sqrt(weight @ simRet.cov() @ weight)
    total_VaR_Pct = -stats.norm.ppf(alpha, loc = total_mu, scale = total_sigma)
    risk_df.loc[m, 'VaR95'] = total_VaR_Pct * total_value
    risk_df.loc[m, 'VaR95_Pct'] = total_VaR_Pct
    risk_df.loc[m, 'ES95_Pct'] = -stats.norm.expect(ub = -risk_df.loc[m, 'VaR95_Pct'], 
                                                    loc = total_mu,
                                                    scale = total_sigma,
                                                    conditional=True)
    risk_df.loc[m, 'ES95'] = risk_df.loc[m, 'ES95_Pct'] * total_value

    return risk_df


def copula_risk(price_path, ptfl_path, nSim, ptfl_name="Total"):
    
    price_df = pd.read_csv(price_path)
    ptfl = pd.read_csv(ptfl_path)
    if (ptfl_name != "Total"):
        ptfl = ptfl[ptfl["Portfolio"] == ptfl_name]

    m, n = price_df.shape
    
    return_df = returns.return_w_method(price_df, "Arithmetic", "Date")
    return_df = return_df.drop(columns=["Date"])
    
    for stock in return_df.columns:
        if stock not in list(ptfl["Stock"]):
            return_df = return_df.drop(columns=[stock])

    # print("PTFL::" ,ptfl)
    ptfl.index = list(ptfl["Stock"])
    for stock in list(ptfl["Stock"]):
        ptfl.loc[stock, "Price"] = price_df.loc[m-1, stock]
        ptfl.loc[stock, "CurrentValue"] = ptfl.loc[stock, "Holding"] * ptfl.loc[stock, "Price"]
        if ptfl.loc[stock, "Portfolio"] == "A" or ptfl.loc[stock, "Portfolio"] == "B":
            ptfl.loc[stock, "Dist"] = "t"
        elif ptfl.loc[stock, "Portfolio"] == "C":
            ptfl.loc[stock, "Dist"] = "normal"

            
    iter = pd.DataFrame({"Iteration": range(nSim)})

    # join tables
    ptfl["key"] = 1
    iter["key"] = 1
    values = pd.merge(ptfl, iter, on="key").drop("key", axis=1)

    # simulate model
    models = {}
    U = pd.DataFrame()

    for stock in return_df.columns:
        if ptfl.loc[stock, "Dist"] == "normal":
            models[stock] = fit.fit_normal(return_df[stock])
        elif ptfl.loc[stock, "Dist"] == "t":
            models[stock] = fit.fit_general_t(return_df[stock])
        
        U[stock] = (return_df[stock] - models[stock].loc[0, "mu"]) / models[stock].loc[0, "sigma"]

    spcor = U.corr(method="spearman")
    uSim = sim.simPca(spcor, nSim)
    uSim = stats.norm.cdf(uSim)
    uSim = pd.DataFrame(uSim, columns = return_df.columns)
    simRet = pd.DataFrame()

    for stock in uSim.columns:
        if ptfl.loc[stock, "Dist"]== "normal":
            simRet[stock] = stats.norm.ppf(uSim[stock], 
                                           loc = models[stock].loc[0, "mu"], 
                                           scale = models[stock].loc[0, "sigma"])
        elif ptfl.loc[stock, "Dist"] == "t":
            simRet[stock] = stats.t.ppf(uSim[stock], 
                                        df = models[stock].loc[0, "nu"], 
                                        loc = models[stock].loc[0, "mu"], 
                                        scale = models[stock].loc[0, "sigma"])
        
    nv = values.shape[0]
    pnl = np.zeros(nv)
    simulatedValue = np.zeros(nv)

    for i in range(nv):
        simulatedValue[i] = values['CurrentValue'][i] * (1 + simRet[values['Stock'][i]][values['Iteration'][i]])
        pnl[i] = simulatedValue[i] - values['CurrentValue'][i]
        
    values['pnl'] = pnl
    values['simulatedValue'] = simulatedValue

    risk_df = aggRisk(simRet, values, models)
    return risk_df