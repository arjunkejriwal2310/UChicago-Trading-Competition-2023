#!/usr/bin/env python
# coding: utf-8

# In[264]:


import numpy as np
import pandas as pd
import scipy as sp
import math
from scipy.optimize import minimize


# In[265]:


# Reading in the asset prices

asset_prices = pd.read_csv("Training Data_Case 3.csv")
asset_prices = asset_prices.drop(columns = asset_prices.columns[0])
asset_prices 


# In[266]:


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


# In[267]:


# Converting the type of data in the returns dataset 

stocks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

for s in stocks:
    returns.loc[:, s] = pd.to_numeric(returns.loc[:,s])

returns.dtypes


# In[268]:


# Creating the training and testing data (to test this optimization method)

training_data = returns.loc[:2000,:].copy() 
testing_data = returns.loc[2001:, :].copy()

testing_data


# In[269]:


# Calculating the covariance matrix for the training data

cov_matrix = pd.DataFrame(np.cov(np.transpose(training_data)))
cov_matrix


# In[270]:


# Finding the weight vector of the last day in the dataset (Just for testing the optimization algorithm)

initial_weights = [0.1 for i in range(0, 10)]
last_return = returns.loc[2518, :]
initial_weights = np.array(initial_weights)
last_return = np.array(last_return)

def portfolio_objective(weights, return_vector, cov_matrix):
    portfolio_return = np.dot(weights, return_vector)
    portfolio_risk = np.matmul(np.matmul(weights, cov_matrix), np.transpose(weights))
    return -(portfolio_return/math.sqrt(portfolio_risk))*math.sqrt(252)

constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
               {'type': 'ineq', 'fun': lambda x: x}]

result = minimize(portfolio_objective, initial_weights, args=(last_return, cov_matrix), method='SLSQP', bounds=[(0, 1)] * returns.shape[1], constraints=constraints)
optimized_weights = result.x
print(optimized_weights)
print(result)
    


# In[271]:


# Function for returning optimal portfolio weights

def return_weights(return_vector, cov_matrix):
    
    initial_weights = [0.1 for i in range(0, 10)]
    initial_weights = np.array(initial_weights)
    return_vector = np.array(return_vector)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
               {'type': 'ineq', 'fun': lambda x: x}]
    
    result = minimize(portfolio_objective, initial_weights, args=(return_vector, cov_matrix), method='SLSQP', bounds=[(0, 1)] * returns.shape[1], constraints=constraints)
    optimized_weights = result.x
    
    # print("The optimized weights are: ", optimized_weights)
    # print("The sharpe ratio is: ", result.fun)
    
    return optimized_weights


# In[ ]:


# Testing the optimization algorithm using the testing data

for i in range(2001,2519):
    training_data.loc[len(training_data.index)] = testing_data.loc[i,:]
    cov_matrix = pd.DataFrame(np.cov(np.transpose(training_data)))
    return_weights(testing_data.loc[i,:],cov_matrix)
    
# We are consistently getting a sharpe ratio of around 1.92, which is a great sign (but we can still try to improve the sharpe ratio)


# In[272]:


# The main function that UChicago will be running

def allocate_portfolio(day_prices):
    
    day_prices = np.array(day_prices)
    asset_prices.loc[len(asset_prices.index)] = day_prices
    # print(asset_prices)
    
    i = len(asset_prices.index)
    j = len(returns.index)
    
    returns.at[j,'A'] = ((asset_prices.at[i-1,'A'] - asset_prices.at[i-2,'A']) / asset_prices.at[i-2,'A'])*100
    returns.at[j,'B'] = ((asset_prices.at[i-1,'B'] - asset_prices.at[i-2,'B']) / asset_prices.at[i-2,'B'])*100
    returns.at[j,'C'] = ((asset_prices.at[i-1,'C'] - asset_prices.at[i-2,'C']) / asset_prices.at[i-2,'C'])*100
    returns.at[j,'D'] = ((asset_prices.at[i-1,'D'] - asset_prices.at[i-2,'D']) / asset_prices.at[i-2,'D'])*100
    returns.at[j,'E'] = ((asset_prices.at[i-1,'E'] - asset_prices.at[i-2,'E']) / asset_prices.at[i-2,'E'])*100
    returns.at[j,'F'] = ((asset_prices.at[i-1,'F'] - asset_prices.at[i-2,'F']) / asset_prices.at[i-2,'F'])*100
    returns.at[j,'G'] = ((asset_prices.at[i-1,'G'] - asset_prices.at[i-2,'G']) / asset_prices.at[i-2,'G'])*100
    returns.at[j,'H'] = ((asset_prices.at[i-1,'H'] - asset_prices.at[i-2,'H']) / asset_prices.at[i-2,'H'])*100
    returns.at[j,'I'] = ((asset_prices.at[i-1,'I'] - asset_prices.at[i-2,'I']) / asset_prices.at[i-2,'I'])*100
    returns.at[j,'J'] = ((asset_prices.at[i-1,'J'] - asset_prices.at[i-2,'J']) / asset_prices.at[i-2,'J'])*100
    
    stocks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    
    for s in stocks:
        returns.loc[:, s] = pd.to_numeric(returns.loc[:,s])
    
    return_vector = returns.loc[len(returns.index)-1, :]
    # print(return_vector)
    
    cov_matrix = pd.DataFrame(np.cov(np.transpose(returns)))
    optimized_weights = return_weights(return_vector,cov_matrix)
    
    return optimized_weights

# Testing the function above

# allocate_portfolio([316, 1315, 202, 286, 53, 95, 156, 360, 90, 35])


# In[273]:


def grading(testing): #testing is a pandas dataframe with price data, index and column names don't matter
    weights = np.full(shape=(len(testing.index),10), fill_value=0.0)
    for i in range(0,len(testing)):
        unnormed = np.array(allocate_portfolio(list(testing.iloc[i,:])))
        positive = np.absolute(unnormed)
        normed = positive/np.sum(positive)
        weights[i]=list(normed)
        #print(weights[i])
    capital = [1]
    for i in range(len(testing) - 1):
        #print(capital[-1] * np.array(weights[i]))
        #print(np.array(testing.iloc[i,:]))
        shares = capital[-1] * np.array(weights[i]) / np.array(testing.iloc[i,:])
        capital.append(float(np.matmul(np.reshape(shares, (1,10)),np.array(testing.iloc[i+1,:]))))
    returns = (np.array(capital[1:]) - np.array(capital[:-1]))/np.array(capital[:-1])
    return np.mean(returns)/ np.std(returns) * (252 ** 0.5), capital, weights


# In[274]:


# asset_prices = pd.read_csv("Training Data_Case 3.csv", index_col = 0)

# grading(asset_prices)

