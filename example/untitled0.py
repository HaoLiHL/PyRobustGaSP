#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:06:14 2023

@author: lihao
"""
import time
import sys
sys.path.append('../')

# import rgasp and functions 
from robustgp import robustgp
from robustgp.src.functions import *
import numpy as np
import scipy as sp
import scipy.stats  
#import lh
num = 1000
start_time = time.time()

a = test_c(num)
end_time = time.time()
print("c++: Total =", a)
print("Time taken:", end_time - start_time, "seconds")

start_time = time.time()

a = test_c(num)
end_time = time.time()
print("c++: Total =", a)
print("Time taken:", end_time - start_time, "seconds")


# Generate a large array of numbers
#numbers = list(range(10000))

# Measure the time it takes to sum the numbers
total = 0
start_time = time.time()

for i in range(num):
    total += 1
end_time = time.time()

# Print the result and the time taken
print("Python: Total =", total)
print("Time taken:", end_time - start_time, "seconds")