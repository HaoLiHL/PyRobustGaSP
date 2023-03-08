#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 21:49:15 2023

@author: lihao
"""
# import rgasp and functions 
import rgasp
from functions import *

import numpy as np
import scipy as sp
import scipy.stats  
import lhsmdu  


P_rgasp = rgasp.rgasp()

###one dimensional test

design = np.array([0.7529149,
7.6520787,
1.9493830,
5.0658406,
3.3174471,
5.8211950,
8.9044846,
6.9543158,
3.4321410,
8.0087184,
0.4770440,
4.0687895,
6.2975229,
9.7600674,
2.6612257]).reshape(-1,1)

response = higdon_1_data(design)
task = P_rgasp.create_task(design, response)  # optimization='nelder-mead'
model = P_rgasp.train(task)
test_input = np.arange(0,10.01,1/100).reshape(-1,1)
result = P_rgasp.predict_rgasp(model, 
                  test_input)


### mutiple dimensional test
design = np.array(lhsmdu.sample(40,8))
response = np.arange(40).reshape(-1,1)

for i in range(40):
    response[i,0] = borehole(design[i,:])

task = P_rgasp.create_task(design, response, method='mmle')  # optimization='nelder-mead'
model = P_rgasp.train(task)

test_input = np.random.normal(size = 80).reshape(10,-1)
result = P_rgasp.predict_rgasp(model, 
                  test_input)
