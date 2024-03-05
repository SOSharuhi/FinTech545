import pandas as pd
import numpy as np

# implement return_calculate() function:
def return_w_method(df, method, date_column):
    
    # drop date column
    prices = df.drop(columns=[date_column])
    
    if method == 'Classical':
        returns = prices.diff().dropna()
    elif method == 'Arithmetic':
        returns = prices.pct_change().dropna()
    elif method == 'Geometric':
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError("Invalid return calculation method specified.")
    
    # Include the date column in the returns DataFrame
    returns[date_column] = df[date_column].iloc[returns.index]
    
    # Reorder columns to have the date column first
    cols = [date_column] + [col for col in returns if col != date_column]
    returns = returns[cols]
    
    return returns