#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np
import pandas as pd
import scipy as sp
import math
from scipy.optimize import minimize


# In[4]:


# Reading in the asset prices

asset_prices = pd.read_csv("Training Data_Case 3.csv")
asset_prices = asset_prices.drop(columns = asset_prices.columns[0])
asset_prices


# In[5]:


# Creating the return dataset

returns = pd.DataFrame(columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])

for i in range(0, 2519):
    returns.at[i,'A'] = ((asset_prices.at[i+1,'A'] - asset_prices.at[i,'A']) / asset_prices.at[i,'A'])*100
    returns.at[i,'B'] = ((asset_prices.at[i+1,'B'] - asset_prices.at[i,'B']) / asset_prices.at[i,'B'])*100
    returns.at[i,'C'] = ((asset_prices.at[i+1,'C'] - asset_prices.at[i,'C']) / asset_prices.at[i,'C'])*100
    returns.at[i,'D'] = ((asset_prices.at[i+1,'D'] - asset_prices.at[i,'D']) / asset_prices.at[i,'D'])*100
    returns.at[i,'E'] = ((asset_prices.at[i+1,'E'] - asset_prices.at[i,'E']) / asset_prices.at[i,'E'])*100
    returns.at[i,'F'] = ((asset_prices.at[i+1,'F'] - asset_prices.at[i,'F']) / asset_prices.at[i,'F'])*100
    returns.at[i,'G'] = ((asset_prices.at[i+1,'G'] - asset_prices.at[i,'G']) / asset_prices.at[i,'G'])*100
    returns.at[i,'H'] = ((asset_prices.at[i+1,'H'] - asset_prices.at[i,'H']) / asset_prices.at[i,'H'])*100
    returns.at[i,'I'] = ((asset_prices.at[i+1,'I'] - asset_prices.at[i,'I']) / asset_prices.at[i,'I'])*100
    returns.at[i,'J'] = ((asset_prices.at[i+1,'J'] - asset_prices.at[i,'J']) / asset_prices.at[i,'J'])*100

returns


# In[23]:


# Converting the type of  

stocks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

for s in stocks:
    returns.loc[:, s] = pd.to_numeric(returns.loc[:,s])

returns.dtypes


# In[25]:


# Calculating the covariance-variance matrix

cov_matrix = pd.DataFrame(np.cov(np.transpose(returns)))
cov_matrix


# In[69]:


# Running optimization on the first day returns 

initial_weights = [0.1 for i in range(0, 10)]
first_return = returns.loc[0, :]
initial_weights = np.array(initial_weights)
first_return = np.array(first_return)

def portfolio_objective(weights, returns, cov_matrix):
    portfolio_return = np.dot(weights, first_return)
    portfolio_risk = np.matmul(np.matmul(weights, cov_matrix), np.transpose(weights))
    return -(portfolio_return/math.sqrt(portfolio_risk))

constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
               {'type': 'ineq', 'fun': lambda x: x}]

result = minimize(portfolio_objective, initial_weights, args=(returns, cov_matrix), method='SLSQP', bounds=[(0, 1)] * returns.shape[1], constraints=constraints)
optimized_weights = result.x
print(optimized_weights)
print(result)

#for i in range(0, 2519):
    #daily_return = returns.loc[i,:]
    

'''
Things to improve:

1) Split into testing and training sets through an 80-20 split (thus your initial covariance matrix will only incorporate 80% of the data)
2) Extend the above solution to all 100% of the data, and create the functionality of reading the data UChicago will provide and calculating the optimal weights   

'''
    


# In[ ]:


# The main function that UChicago will be running

def allocate_portfolio(asset_prices):
    
    # This simple strategy equally weights all assets every period
    # (called a 1/n strategy).
    
    n_assets = len(asset_prices)
    weights = np.repeat(1 / n_assets, n_assets)
    return weights

