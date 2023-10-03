#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 21:49:15 2023

@author: lihao
"""

import sys
#sys.path.append('../')

# import PyRobustGaSP and functions 
import PyRobustGaSP
from src.functions import *
import numpy as np
import scipy as sp
import scipy.stats  
import lhsmdu  


P_rgasp = PyRobustGaSP.PyRobustGaSP()


##1D function

from scipy.stats import qmc
import matplotlib.pyplot as plt


sampler = qmc.LatinHypercube(d=1)
sample_input = 10 * sampler.random(n=15)
sample_output = higdon_1_data(sample_input)

task = P_rgasp.create_task(sample_input, sample_output)  # optimization='nelder-mead'
model = P_rgasp.train_rgasp(task)

testing_input = np.arange(0,10,1/100).reshape(-1,1)
testing_predict = P_rgasp.predict_rgasp(model, 
                  testing_input)

testing_output=higdon_1_data(testing_input)


# Display the plot
fig, ax = plt.subplots()
# Plot the first line on the axes
ax.plot(testing_input,testing_predict['mean'], label='Predicted', color = 'blue')

# Plot the second line on the axes
ax.plot(testing_input, testing_output, color = 'yellow', label='real')
ax.fill_between(testing_input[:,0], testing_predict['upper95'], testing_predict['lower95'], alpha=0.2)
# Set the labels and title of the plot
ax.set_xlabel('input')
ax.set_ylabel('output')
#ax.set_title('Sine and Cosine Waves')
# Add a legend to the plot
ax.legend()
plt.show()



#######friedman function

sampler = qmc.LatinHypercube(d=5)
sample_input = sampler.random(n=40)

from numpy import genfromtxt
#my_sample_inputdata = genfromtxt('/Users/HL/Desktop/input.csv', delimiter=',')
#sample_input = my_sample_inputdata[1:,]
num_obs=sample_input.shape[0]
sample_output= np.zeros((num_obs,1))

for i in range(num_obs):
    sample_output[i,0]=friedman_5_data(sample_input[i,:])

task = P_rgasp.create_task(sample_input, sample_output)  # optimization='nelder-mead'
model = P_rgasp.train_rgasp(task)


dim_inputs=sample_input.shape[1]
num_testing_input = 200    
testing_input =  np.random.uniform(size =num_testing_input*dim_inputs).reshape(num_testing_input,-1)

testing_predict = P_rgasp.predict_rgasp(model, 
                  testing_input)

testing_output = np.zeros((num_testing_input,1))

for i in range(num_testing_input):
    testing_output[i,0]=friedman_5_data(testing_input[i,:])


m_rmse=np.sqrt(np.mean( (testing_predict['mean']-testing_output[:,0])**2))#/np.std(testing_output[:,0])
print('RMSE is ', m_rmse)

# Display the plot
fig, ax = plt.subplots()
# Plot the first line on the axes
ax.scatter(testing_predict['mean'],testing_output, label='Predicted', color = 'blue')
ax.set_xlabel('prediction')
ax.set_ylabel('real output')
ref_line_x = np.array([5, 25])
ref_line_y = np.array([5, 25])
ax.plot(ref_line_x, ref_line_y, linestyle='--', color='r')
plt.show()

#####see the proportion 
prop_m = np.sum((testing_predict['lower95']<=testing_output[:,0]) & (testing_predict['upper95']>=testing_output[:,0]))/testing_output.shape[0]
print("The Proportion of the test output covered by 95% CI is ",prop_m)

length_m = np.mean(testing_predict['upper95']-testing_predict['lower95'])
print("The average length of the CI is ",length_m)


trend_rgasp = np.column_stack((np.repeat(1.0,num_obs),sample_input))
task_trend = P_rgasp.create_task(sample_input, sample_output,trend = trend_rgasp)  # optimization='nelder-mead'
model_trend = P_rgasp.train_rgasp(task_trend)

trend__test_rgasp = np.column_stack((np.repeat(1.0,num_testing_input),testing_input))

testing_trend_predict = P_rgasp.predict_rgasp(model_trend, 
                  testing_input, testing_trend = trend__test_rgasp)

m_trend_rmse=np.sqrt(np.mean( (testing_trend_predict['mean']-testing_output[:,0])**2))
print('RMSE is ', m_trend_rmse)
# Display the plot
fig, ax = plt.subplots()
# Plot the first line on the axes
ax.scatter(testing_trend_predict['mean'],testing_output, label='Predicted', color = 'blue')
ax.set_xlabel('prediction')
ax.set_ylabel('real output')
ref_line_x = np.array([5, 25])
ref_line_y = np.array([5, 25])
ax.plot(ref_line_x, ref_line_y, linestyle='--', color='r')
plt.show()

prop_m = np.sum((testing_trend_predict['lower95']<=testing_output[:,0]) & (testing_trend_predict['upper95']>=testing_output[:,0]))/testing_output.shape[0]
print("The Proportion of the test output covered by 95% CI is ",prop_m)

length_m = np.mean(testing_trend_predict['upper95']-testing_trend_predict['lower95'])
print("The average length of the CI is ",length_m)




#####  PP GaSP Emulation
from numpy import genfromtxt
import pandas as pd

humanity_X = genfromtxt('../PyRobustGaSP/src/dataset/humanity_X.csv', delimiter=',')[1:,:]
humanity_Y = genfromtxt('../PyRobustGaSP/src/dataset/humanity_Y.csv', delimiter=',')[1:,:]
task = P_rgasp.create_task(humanity_X, humanity_Y, nugget_est=True, num_initial_values = 3)  # optimization='nelder-mead'
model = P_rgasp.train_ppgasp(task)
humanity_Xt = genfromtxt('../PyRobustGaSP/src/dataset/humanity_Xt.csv', delimiter=',')[1:,:]
humanity_Yt = genfromtxt('../PyRobustGaSP/src/dataset/humanity_Yt.csv', delimiter=',')[1:,:]

testing_predict = P_rgasp.predict_ppgasp(model, 
                  humanity_Xt)

m_rmse=np.sqrt(np.mean( (testing_predict['mean']-humanity_Yt)**2))#/np.std(testing_output[:,0])
print('RMSE is ', m_rmse)
print("std of testing y", np.std(humanity_Yt))

prop_m = np.sum((testing_predict['lower95']<=humanity_Yt) & (testing_predict['upper95']>=humanity_Yt))/(humanity_Yt.shape[0]*humanity_Yt.shape[1])
print("The Proportion of the test output covered by 95% CI is ",round(prop_m,4))

#### ppgasp with trend 
humanity_X2 = pd.read_csv('../PyRobustGaSP/src/dataset/humanity_X_2.csv')
humanity_Xt2 = pd.read_csv('../PyRobustGaSP/src/dataset/humanity_Xt_2.csv')

trend_ppgasp = np.column_stack((np.repeat(1.0,humanity_X.shape[0]),humanity_X2['foodC'].to_numpy()))
task_trend = P_rgasp.create_task(humanity_X, humanity_Y,trend = trend_ppgasp, nugget_est=True, num_initial_values = 3)  # optimization='nelder-mead'
model_trend = P_rgasp.train_ppgasp(task_trend)

trend__test_rgasp = np.column_stack((np.repeat(1.0,humanity_Xt.shape[0]),humanity_Xt2['foodC'].to_numpy()))

testing_trend_predict = P_rgasp.predict_ppgasp(model_trend, 
                  humanity_Xt, testing_trend = trend__test_rgasp)

m_rmse=np.sqrt(np.mean( (testing_trend_predict['mean']-humanity_Yt)**2))#/np.std(testing_output[:,0])
print('RMSE is ', m_rmse)
print("std of testing y", np.std(humanity_Yt))
prop_m = np.sum((testing_trend_predict['lower95']<=humanity_Yt) & (testing_trend_predict['upper95']>=humanity_Yt))/(humanity_Yt.shape[0]*humanity_Yt.shape[1])
print("The Proportion of the test output covered by 95% CI is ",round(prop_m,4))


# #### TITAN2D data
# import pandas as pd
# input_variables = pd.read_csv('/Users/HL/Documents/GitHub/P_RobustGP/dataset/input_variables.csv',index_col =0).to_numpy()
# pyroclastic_flow_heights = pd.read_csv('/Users/HL/Documents/GitHub/P_RobustGP/dataset/pyroclastic_flow_heights.csv',index_col =0).to_numpy()
# loc_index = pd.read_csv('/Users/HL/Documents/GitHub/P_RobustGP/dataset/loc_index.csv',index_col =0).to_numpy()

# #loc_index.reset_index(inplace = True, drop = True)

# #pyroclastic_flow_heights.reset_index(inplace = True)
# training_input = input_variables[0:50,:]
# testing_input = input_variables[50:683,:]

# output=pyroclastic_flow_heights[0:50,(loc_index[2,]==1)]
# testing_output=pyroclastic_flow_heights[50:683,loc_index[2,]==1]


# n = output.shape[0]
# n_testing = testing_output.shape[0]

# ##delete those location where all output are zero

# index_all_zero=[]
# for i in range(output.shape[1]):
#     if np.sum(output[:,i]==0)==50:
#         index_all_zero.append(i)

# output_log=np.log(output+1)
# output_log_1 = np.delete(output_log,index_all_zero, axis = 1)
# k=output_log_1.shape[1]

# trend = np.column_stack((np.repeat(1.0,n),training_input[:,0]))
# task = P_rgasp.create_task(training_input[:,0:3], output_log_1, trend = trend, nugget_est=True, max_eval = 100,num_initial_values = 3)  # optimization='nelder-mead'
# model = P_rgasp.train_ppgasp(task)

# testing_trend_predict = P_rgasp.predict_ppgasp(model, 
#                   testing_input[:,0:3], testing_trend = np.column_stack((np.repeat(1.0,n_testing),testing_input[:,0])))

# m_pred_ppgasp_mean=np.exp(testing_trend_predict['mean'])-1
# m_pred_ppgasp_LB=np.exp(testing_trend_predict['lower95'])-1
# m_pred_ppgasp_UB=np.exp(testing_trend_predict['upper95'])-1

# test_out = np.delete(testing_output,index_all_zero, axis = 1)
# m_trend_rmse=np.sqrt(np.mean( (m_pred_ppgasp_mean-np.delete(testing_output,index_all_zero, axis = 1))**2))
# print('RMSE is ', m_trend_rmse)
# prop_m = np.sum((m_pred_ppgasp_LB<=test_out) & (m_pred_ppgasp_UB>=test_out))/(test_out.shape[0]*test_out.shape[1])
# print("The Proportion of the test output covered by 95% CI is ",round(prop_m,4))





###one dimensional test

# design = np.array([0.7529149,
# 7.6520787,
# 1.9493830,
# 5.0658406,
# 3.3174471,
# 5.8211950,
# 8.9044846,
# 6.9543158,
# 3.4321410,
# 8.0087184,
# 0.4770440,
# 4.0687895,
# 6.2975229,
# 9.7600674,
# 2.6612257]).reshape(-1,1)

# response = higdon_1_data(design)
# task = P_rgasp.create_task(design, response)  # optimization='nelder-mead'
# model = P_rgasp.train(task)
# test_input = np.arange(0,10.01,1/100).reshape(-1,1)
# result = P_rgasp.predict_rgasp(model, 
#                   test_input)


### mutiple dimensional test
# design = np.array(lhsmdu.sample(40,8))
# response = np.arange(40).reshape(-1,1)

# for i in range(40):
#     response[i,0] = borehole(design[i,:])

# task = P_rgasp.create_task(design, response, method='mmle')  # optimization='nelder-mead'
# model = P_rgasp.train(task)

# test_input = np.random.normal(size = 80).reshape(10,-1)
# result = P_rgasp.predict_rgasp(model, 
#                   test_input)
