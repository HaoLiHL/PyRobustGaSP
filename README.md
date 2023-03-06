# P_RobustGP
Python version of robust GP

## Installation

1. need install Python 3.7+, Pybind11, cppimports (see https://pybind11.readthedocs.io/en/latest/)
2. need install eigen,This link may be helpful for MAC users: https://stackoverflow.com/questions/35658420/installing-eigen-on-mac-os-x-for-xcode
3. then replace the setting line in my_own.cpp: cfg['include_dirs'] = ['your_path_of_eigen'] (e.g ['Users/HL/Downloads/eigen-3.4.0']) 
