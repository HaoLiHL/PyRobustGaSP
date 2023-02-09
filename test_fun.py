#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 21:13:30 2023

@author: lihao
"""
import cppimport
import numpy as np

code = cppimport.imp("my_own")

if __name__ == '__main__':
    A = np.array([[1,2,1],
                  [2,1,0.1],
                  [1,0.1,2]])
    A = np.array([[1,2],
                  [2,1]])

    #print(A)
    #print(code.matern_5_2_funct(A,0.1))
    #print(code.periodic_gauss_funct(A,0.1))
    #print(code.periodic_exp_funct_fixed_normalized_const(A,0.1,0.1))
    #print(code.separable_kernel([A,A],np.array([0.1,0.1]),'matern_3_2',np.array([0.1,0.1])))
    #print(code.separable_multi_kernel_pred_periodic([A,A],np.array([0.1,0.1]),[1,2],np.array([0.1,0.1]),np.array([0.11,0.11])))
    #print(code.euclidean_distance(A,A))
    
   # print(code.log_profile_lik(np.array([0.1,0.1]),0.1, True,[A,A],A,"Yes",A,np.array([1,2]),np.array([0.1,0.1])))
    #print(code.log_approx_ref_prior(np.array([0.1,0.1]),0.1, True,np.array([0.1,0.1]),0.1,0.1))
    #print(code.log_marginal_lik_deriv(np.array([0.1,0.1]),0.1, True,[A,A],A,"Yes",A,np.array([1,2]),np.array([0.1,0.1])))
    #print(code.log_profile_lik_deriv(np.array([0.1,0.1]),0.1, True,[A,A],A,"Yes",A,np.array([1,2]),np.array([0.1,0.1])))

    print(code.log_approx_ref_prior_deriv(np.array([0.1,0.1]),0.1, True,np.array([0.1,0.1]),0.1,0.2))
    print(code.log_profile_lik_deriv(np.array([0.1,0.1]),0.1, True,[A,A],A,"Yes",A,np.array([1,2]),np.array([0.1,0.1])))

# log_ref_marginal_post(const Eigen::VectorXd & param,double nugget, 
#                       bool nugget_est, const py::list& R0, const Eigen::MatrixXd & X,
#                       const std::string zero_mean,const Eigen::MatrixXd & output,
#                       const Eigen::VectorXi & kernel_type,const Eigen::VectorXd & alpha){

# log_approx_ref_prior_deriv(const Eigen::VectorXd & param,double nugget, bool nugget_est, const Eigen::VectorXd & CL,const double a,const double b ){
    

# import cppimport
# import numpy as np

    
# my_own = cppimport.imp("my_own")

# if __name__ == '__main__':
#     xs = np.random.rand(10)
#     print(xs)
#     print(my_own.square(xs))

#     #ys = range(10)
#     #print(my_own.square(ys))
