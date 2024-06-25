#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 23:11:51 2023

@author: lihao
"""

from PyRobustGaSP import PyRobustGaSP
from src.functions import *

import numpy as np
import scipy as sp
import scipy.stats 
from scipy.stats import qmc
import matplotlib.pyplot as plt


def branin(x):
    x1 = x[0]
    x2 = x[1]
    t = 1/(8*np.pi)
    ans = 1*(x2-5.1/(np.pi**2 *4)*x1**2+5/np.pi*x1-6)**2+10*(1-t)*np.cos(x1)+10
    return ans

def rescale(arr,x1_range,x2_range):
    # rescale the samples from  [0,1]*[0,1] to x1_range * x2_range
    arr[:,0] = x1_range[0] + arr[:,0] * (x1_range[1] - x1_range[0])
    arr[:,1] = x2_range[0] + arr[:,1] * (x2_range[1] - x2_range[0])
    return arr






#-----------rgasp 2d ackleu-----------
def ackley(x):
    ans = -20*np.exp(-0.2*np.sqrt(0.5*(x[0]**2 + x[1]**2)))
    -np.exp(0.5*(np.cos(np.pi*2*x[1]+np.pi*2*x[0])))+np.exp(1)+20
    return ans
# rescale the input space 
def rescale(arr,x1_range,x2_range):
    # rescale the samples from  [0,1]*[0,1] to x1_range * x2_range
    arr[:,0] = x1_range[0] + arr[:,0] * (x1_range[1] - x1_range[0])
    arr[:,1] = x2_range[0] + arr[:,1] * (x2_range[1] - x2_range[0])
    return arr

# Create a PyRobustGaSP model instance
P_rgasp = PyRobustGaSP()

# Generate the training sample
sampler = qmc.LatinHypercube(d=2)
sample_input = sampler.random(n=80)
x1_range = (-5, 5)
x2_range = (-5, 5)
sample_input = rescale(sample_input,x1_range,x2_range)
num_obs=sample_input.shape[0]
sample_output= np.zeros((num_obs,1))
for i in range(num_obs):
    sample_output[i,0]=ackley(sample_input[i,:])

noise = np.random.normal(0, 1e-3, sample_output.shape)
sample_output = sample_output+noise

# Create a task for model training
task = P_rgasp.create_task(sample_input, sample_output, nugget_est=True)  
# Fit a rgasp model using created task
model = P_rgasp.train_rgasp(task)

# Get testing input and output
dim_inputs=sample_input.shape[1]
num_testing_input = 200    
testing_input = np.random.uniform(size =num_testing_input*
                                    dim_inputs).reshape(num_testing_input,-1)
testing_input = rescale(testing_input,x1_range,x2_range)

testing_output = np.zeros((num_testing_input,1))
for i in range(num_testing_input):
    testing_output[i,0]=ackley(testing_input[i,:])

# Get the rgasp predict object
testing_predict = P_rgasp.predict_rgasp(model, 
                  testing_input)


ratio_rmse_std=np.sqrt(np.mean( (testing_predict['mean']-testing_output[:,0])**2))/np.std(testing_output)
print('RMSE/STD is ', ratio_rmse_std)