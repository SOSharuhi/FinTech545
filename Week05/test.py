import RMLib.covar as covar
import RMLib.non_psd as non_psd
import RMLib.returns as returns
import RMLib.simulation as sim
import RMLib.value_at_risk as var
import RMLib.expected_shortfall as es
import RMLib.fit_model as fit
import RMLib.copula as copula

import pandas as pd
import numpy as np
from scipy import stats

from inspect import getmembers, isfunction
import RMLib
    
# show all functions created

def show_functions(name, module):
    print(name)
    functions_list = getmembers(module, isfunction)
    for func in functions_list:
        print(func[0])
    print(" ")

show_functions("1. Covariance & Correlation:", covar)
show_functions("2. Non-PSD Fixes: ", non_psd)
show_functions("3. Simulation Methods: ", sim)
show_functions("4. Return Calculation: ", returns)
show_functions("5. Parametric Models: ", fit)
show_functions("6. VaR Calculation: ", var)
show_functions("7. Expected Shortfall Calculation: ", es)
show_functions("8. Copula: ", copula)

## Question 1

# test function
def test(cout, filename, precision):
    filepath = 'data/' + filename + '.csv'
    df = pd.read_csv(filepath)
    diff = cout.reset_index(drop=True) - df.reset_index(drop=True)
    tol = 0.1 ** precision
    exceed_tol = diff >= tol
    print(filename, " " , exceed_tol.sum().sum() == 0)
    
def test1(cout, filename, precision):
    filepath = 'data/' + filename + '.csv'
    df = pd.read_csv(filepath)
    cout = cout.reset_index(drop=True).round(precision)
    # print(cout)
    df = df.reset_index(drop=True).round(precision)
    # print(df)
    print(filename, " ", cout.equals(df))

# Test 1 - missing covariance calculations
# Generate some random numbers with missing values.

x = pd.read_csv('data/test1.csv')
# 1.1 Skip Missing rows - Covariance
cout = covar.Cov(x)
test(cout, 'testout_1.1', 9)
# 1.2 Skip Missing rows - Correlation
cout = covar.Cor(x)
test(cout, 'testout_1.2', 9)
# 1.3 Pairwise - Covariance
cout = covar.Cov(x, False)
test(cout, 'testout_1.3', 9)
# 1.4 Pairwise - Correlation
cout = covar.Cor(x, False)
test(cout, 'testout_1.4', 9)

# Test 2 - EW Covariance
x = pd.read_csv("data/test2.csv")
# 2.1 EW Covariance λ=0.97
lam = 0.97
cout = covar.ewCovar(x, lam)
test(cout, 'testout_2.1', 9)
# 2.2 EW Correlation λ=0.94
lam = 0.94
cout = covar.ewCorr(x, lam)
test(cout, 'testout_2.2', 9)
# 2.3 EW Cov w/ EW Var(λ=0.94) EW Correlation(λ=0.97)
cout = covar.ewCovCor(x, 0.97, 0.94)
test(cout, 'testout_2.3', 9)

# Test 3 - non-psd matrices
x = pd.read_csv("data/testout_1.3.csv")
# 3.1 near_psd covariance
cout = non_psd.nearPSDCov(x)
test(cout, 'testout_3.1', 9)
# 3.2 near_psd Correlation
x = pd.read_csv("data/testout_1.4.csv")
cout = non_psd.nearPSDCor(x)
test(cout, 'testout_3.2', 9)
# 3.3 Higham covariance
x = pd.read_csv("data/testout_1.3.csv")
cout = non_psd.higham_nearestPSDCov(x)
test(cout, 'testout_3.3', 9)
# 3.4 Higham Correlation
x = pd.read_csv("data/testout_1.4.csv")
cout = non_psd.higham_nearestPSDCor(x)
test(cout, 'testout_3.4', 9)

# Test 4 - cholesky factorization
x = pd.read_csv('data/testout_3.1.csv')
cout = non_psd.chol_psd(x)
test(cout, 'testout_4.1', 6)

# Test 5 - Normal Simulation

# 5.1 PD Input
x = pd.read_csv('data/test5_1.csv')
cout = covar.Cov(sim.simNormal(x, 100000))
test(cout, 'testout_5.1', 3)
# txt = " Cannot compare simulation results"
# print('testout_5.1', txt)
# print('Simulated Cov Mat:')
# print(cout)
# 5.2 PSD Input
x = pd.read_csv('data/test5_2.csv')
cout = covar.Cov(sim.simNormal(x, 100000))
test(cout, 'testout_5.2', 3)
# 5.3 nonPSD Input, near_psd fix
x = pd.read_csv('data/test5_3.csv')
cout = covar.Cov(sim.simNormal(x, 100000, 'near_psd'))
test(cout, 'testout_5.3', 3)
# 5.4 nonPSD Input Higham Fix
x = pd.read_csv('data/test5_3.csv')
cout = covar.Cov(sim.simNormal(x, 100000, 'higham_nearestPSD'))
test(cout, 'testout_5.4', 3)
# 5.5 PSD Input - PCA Simulation
x = pd.read_csv('data/test5_2.csv')
cout = covar.Cov(sim.simPca(x, 100000, 0.99))
test(cout, 'testout_5.5', 3)

# Test 6 - Returns
price = pd.read_csv("data/test6.csv")
# 6.1 Arithmetic returns
rout = returns.return_w_method(price, "Arithmetic", "Date")
test1(rout, 'test6_1', 9)
# 6.2 Log returns
rout = returns.return_w_method(price, "Geometric", "Date")
test1(rout, 'test6_2', 9)

# Test 7 - Fit Distribution
# Data simulation

# 7.1 Fit Normal Distribution
x = pd.read_csv("data/test7_1.csv")
cout = fit.fit_normal(x)
test1(cout, 'testout7_1', 3)
# 7.2 Fit TDist
x = pd.read_csv("data/test7_2.csv")
cout = fit.fit_general_t(x)
test1(cout, 'testout7_2', 4)
# 7.3 Fit T Regression
x = pd.read_csv("data/test7_3.csv")
cout = fit.fit_regression_t(x)
test1(cout, 'testout7_3', 3)

# Test 8 - VaR

# 8.1 VaR Normal
x = pd.read_csv("data/test7_1.csv")
cout = var.VaR_normal_distribution(x)
test1(cout, 'testout8_1', 3)

# 8.2 VaR TDist
x = pd.read_csv("data/test7_2.csv")
cout = var.VaR_t_distribution(x)
test1(cout, 'testout8_2', 3)

# 8.3 VaR Simulation
x = pd.read_csv("data/test7_2.csv")
cout = var.VaR_simulation(x)
test1(cout, 'testout8_3', 2)

# 8.4 ES Normal
x = pd.read_csv("data/test7_1.csv")
cout = es.ES_normal_distribution(x)
test1(cout, 'testout8_4', 3)

# 8.5 ES TDist
x = pd.read_csv("data/test7_2.csv")
cout = es.ES_t_distribution(x)
test1(cout, 'testout8_5', 3)

# 8.6 VaR Simulation
x = pd.read_csv("data/test7_2.csv")
cout = es.ES_simulation(x)
test1(cout, 'testout8_6', 2)

# Test 9

# 9.1

df_return = pd.read_csv("data/test9_1_returns.csv")

prices = {
    "A": 20.0,
    "B": 30.0
}

models = {
    "A": fit.fit_normal(df_return["A"]),
    "B": fit.fit_general_t(df_return["B"])
}

nSim = 100000
muA = models["A"]["mu"].to_numpy()
muB = models["B"]["mu"].to_numpy()
sigmaA = models['A']['sigma'].to_numpy()
sigmaB = models['B']['sigma'].to_numpy()
dfB = models['B']['nu'].to_numpy()

U = pd.DataFrame({
    "A": (df_return["A"] - muA) / sigmaA,
    "B": (df_return["B"] - muB) / sigmaB
})

spcor = U.corr(method="spearman")
# np.random.seed(4)
uSim = sim.simPca(spcor, nSim)
uSim = stats.norm.cdf(uSim)

simRet = pd.DataFrame({
    "A": stats.norm.ppf(uSim[:,0], loc = muA, scale = sigmaA),
    "B": stats.t.ppf(uSim[:,1], df = dfB, loc = muB, scale = sigmaB)
})

portfolio = pd.DataFrame({
    "Dist": ["normal", "t"],
    "Stock": ["A", "B"],
    "CurrentValue": [2000, 3000]
})

# Generating the iteration DataFrame
iteration = pd.DataFrame({'Iteration': range(nSim)})

# Performing the cross join in pandas
# In pandas, a cross join can be performed by temporarily assigning a common key and merging on it
portfolio['key'] = 1
iteration['key'] = 1
values = pd.merge(portfolio, iteration, on='key').drop('key', axis=1)

nv = values.shape[0]
pnl = np.zeros(nv)
simulatedValue = np.zeros(nv)
i = 0

for i in range(nv):
    simulatedValue[i] = values['CurrentValue'][i] * (1 + simRet[values['Stock'][i]][values['Iteration'][i]])
    pnl[i] = simulatedValue[i] - values['CurrentValue'][i]

values['pnl'] = pnl
values['simulatedValue'] = simulatedValue
# print("//////")
# print(values)
# print(models)

risk_df = copula.aggRisk(simRet, values, models)

print(risk_df)


# Question 2
print("")
print("Question 2:")
print(" ")
x = pd.read_csv('problem1.csv')

# VaR EW normal
print("VaR EW Normal w/ Lambda = 0.97")
lam = 0.97
cout = var.Var_normal_distribution_EW(x, lam)
print(cout)

# VaR general T
print("VaR general T | MLE T")
cout = var.VaR_t_distribution(x)
print(cout)

# VaR Historic
print("VaR Historic")
cout = var.VaR_historic(x)
print(cout)

# ES EW normal
print("ES EW Normal w/ Lambda = 0.97")
cout = es.ES_normal_distribution_EW(x, lam)
print(cout)

# ES general T (MLE)
print("ES general T | MLE T")
cout = es.ES_t_distribution(x)
print(cout)

# ES Historic
print("ES Historic")
cout = es.ES_historic(x)
print(cout)

print(" ")
# Question 3

# price dataframe
price_path = "DailyPrices.csv"

# portfolio dataframe
ptfl_path = "portfolio.csv"

nSim = 10000

print("Portfolio A:")
df = copula.copula_risk(price_path, ptfl_path, nSim, "A")
print(df[df["Stock"] == "Total"])
print("")

print("Portfolio B:")
df = copula.copula_risk(price_path, ptfl_path, nSim, "B")
print(df[df["Stock"] == "Total"])
print("")

print("Portfolio C:")
df = copula.copula_risk(price_path, ptfl_path, nSim, "C")
print(df[df["Stock"] == "Total"])
print("")

print("Total:")
df = copula.copula_risk(price_path, ptfl_path, nSim)
print(df[df["Stock"] == "Total"])
print("")

