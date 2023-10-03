#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 23:08:02 2023

@author: lihao
"""
import sys
sys.path.append('../')

from robustgp import robustgp
from robustgp.src.functions import *

import numpy as np
import scipy as sp
import pandas as pd
import scipy.stats 
from scipy.stats import qmc
import matplotlib.pyplot as plt
import lhsmdu 
from numpy import genfromtxt


# P_rgasp = robustgp.robustgp()

# X_train = np.genfromtxt("/Users/HL/Documents/GitHub/pyrgasp/dataset/X_train.txt", skip_header=1)[0:50,1:]
# Y_train =np.genfromtxt("/Users/HL/Documents/GitHub/pyrgasp/dataset/Y_train.txt", skip_header=1)[0:50,1:]
# task = P_rgasp.create_task(X_train, Y_train,isotropic=True,initial_values=[-12,-8])  # optimization='nelder-mead'

# model = P_rgasp.train_ppgasp(task)