# pyrgasp

Python version of robust GaSP and PPGaSP

## Requiements
Python 3.7+, Numpy (>=1.19), Scipy (>=1.1), Pybind11, cppimport (see https://pybind11.readthedocs.io/en/latest/)

## Getting Started

Just need to decompressed the robustgp.zip. More examples are shown in example/example.py

## Example: Build a robust GaSP model  
```python
import numpy as np
from robustgp import robustgp
from robustgp.src.functions import *
from scipy.stats import qmc


P_rgasp = robustgp.robustgp()

##1D function

sampler = qmc.LatinHypercube(d=1)
sample_input = 10 * sampler.random(n=15)
sample_output = higdon_1_data(sample_input)


# Create a task for pyrgasp
task = P_rgasp.create_task(sample_input, sample_output) 
# Train a rgasp model 
model = P_rgasp.train_rgasp(task)

testing_input = np.arange(0,10,1/100).reshape(-1,1)

# Get the prediction dict using fitted rgasp model
testing_predict = P_rgasp.predict_rgasp(model, 
                  testing_input)

testing_output=higdon_1_data(testing_input)

```



## Authors:
Hao Li, Mengyang Gu

## References
1. Gu, M. & Berger, J. (2016). Parallel partial Gaussian process emulation for computer
models with massive output. The Annals of Applied Statistics 10(3), 1317-1347.
2. Gu, M., Wang, X. & Berger, J. (2018). Robust Gaussian stochastic process emulation. The
Annals of Statistics. 46(6A), 3038-3066.
3. Gu, M. (2019). Jointly robust prior for Gaussian stochastic process in emulation,
calibration and variable selection. Bayesian Analysis. 14(3), 857-885.
