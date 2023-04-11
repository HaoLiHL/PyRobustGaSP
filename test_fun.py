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
    A = np.array([[0.1,0.2],
                  [0.2,0.1]])
    X = np.array([[0.1,0.2],
                  [0.3,0.4]])
    L = np.array([[0.1,0],
                  [0.2,0.1]])
    
    # print(code.log_marginal_lik_ppgasp(np.array(1,2), 0.1, True, [A,A+0.1], X,
    #                                    "Yes", ))
    # log_marginal_lik_ppgasp(const Eigen::VectorXd &  param,double nugget, const bool nugget_est,
    #                         const py::list& R0, const Eigen::MatrixXd & X,
    #                         const std::string zero_mean,const Eigen::MatrixXd & output, 
    #                         const Eigen::VectorXi  &kernel_type,const Eigen::VectorXd & alpha ){

                                

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
    print(code.log_approx_ref_prior_deriv(np.array([1.31]),0.1, False,np.array([0]),0.2,0.08))
    #print(code.log_approx_ref_prior_deriv(np.array([0.1,0.1]),0.1, True,np.array([0.1,0.1]),0.1,0.2))
    #print(code.log_profile_lik_deriv(np.array([0.1,0.1]),0.1, True,[A,A],A,"Yes",A,np.array([1,2]),np.array([0.1,0.1])))
    #above checked 
    #print(code.my_test(np.array([0.1,0.1]),A))
    ##print(code.log_ref_marginal_post(np.array([0.1,0.1]),0.1, True,[A,A],A,"Yes",X,np.array([1,2]),np.array([0.1,0.1])))
#log_ref_marginal_post
    #print(code.construct_rgasp(np.array([0.1,0.1]),0.1,[A,A], A,"Yes",A,np.array([1,2]),np.array([0.1,0.1])))
    
    
    #print(code.pred_rgasp(np.array([0.1,0.1]),0.1, A,A,"Yes",A,A,A,A,L,L,0.1,0.1,0.9,[A,A],np.array([1,2]),np.array([0.1,0.1]),"post_mode",True))
    # print(code.generate_predictive_mean_cov(np.array([0.1,0.1]),0.1, A,
    #                                         A,"Yes",A,A,A,A,A,np.array([0.1,0.1]),0.1,
    #                                         [A,A],[A,A],np.array([1,2]),
    #                                         np.array([0.1,0.1]),"post_mode",True))
    
    # print(code.log_profile_lik_ppgasp(np.array([0.1,0.1]),0.1, True,[A,A],A,'Yes',X,np.array([1,2]),
    #                                         np.array([0.1,0.1])))
    
    # print(code.log_ref_marginal_post_ppgasp(np.array([0.1,0.1]),0.1, True,[A,A],A,'Yes',X,np.array([1,2]),
    #                                         np.array([0.1,0.1])))
    
    # print(code.construct_ppgasp(np.array([0.1,0.1]),0.1, [A,A],A,'Yes',X,np.array([1,2]),
    #                                         np.array([0.1,0.1])))
    
    # print(code.pred_ppgasp(np.array([0.1,0.1]),0.1, A,
    #                                         A,"Yes",A,A,A,A,A,A,np.array([0.1,0.1]),
    #                                         0.1,0.9,[A,A],np.array([1,2]),
    #                                         np.array([0.1,0.1]),"post_mode",True))
    
    #print(code.test_const_column(A))
    
    
    #bool test_const_column (const Eigen::MatrixXd &d){
    
    # py::list pred_ppgasp(const Eigen::VectorXd beta,const double nu, const  pred_ppgasp & input,  
    #                      const pred_ppgasp & X,const  std::string zero_mean, const pred_ppgasp & output,
    #                      const pred_ppgasp & testing_input, const pred_ppgasp & X_testing, 
    #                  const pred_ppgasp & L ,const pred_ppgasp & LX, const pred_ppgasp & theta_hat,  
    #                  const Eigen::VectorXd &  sigma2_hat,double q_025, double q_975, py::list r0,
    #                  Eigen::VectorXi kernel_type,const Eigen::VectorXd alpha, const  std::string method, 
    #                  const bool interval_data){

    # py::list construct_ppgasp(const Eigen::VectorXd & beta,const double nu,  const py::list R0, 
    #                           const Eigen::MatrixXd & X,const  std::string zero_mean,const Eigen::MatrixXd & output,
    #                           const Eigen::VectorXi & kernel_type,const Eigen::VectorXd & alpha){

    
    
    # Eigen::VectorXd log_marginal_lik_deriv_ppgasp(const Eigen::VectorXd & param,double nugget,  
    #                                               bool nugget_est, const py::list R0, 
    #                                               const Eigen::MatrixXd & X,const std::string zero_mean,
    #                                               const Eigen::MatrixXd & output, const Eigen::VectorXi & kernel_type,
    #                                               const Eigen::VectorXd & alpha){
      

# double log_profile_lik_ppgasp(const Eigen::VectorXd &   param,double nugget, 
#                               const bool nugget_est, const py::list& R0, 
#                               const Eigen::MatrixXd & X,const std::string zero_mean,
#                               const Eigen::MatrixXd & output,const Eigen::VectorXi &kernel_type,
#                               const Eigen::VectorXd &alpha ){

# double log_marginal_lik_ppgasp(const Eigen::VectorXd &  param,double nugget, const bool nugget_est, 
#                                const py::list& R0, const Eigen::MatrixXd & X,const std::string zero_mean,
#                                const Eigen::MatrixXd & output, const Eigen::VectorXi  &kernel_type,const Eigen::VectorXd & alpha ){

# generate_predictive_mean_cov(const Eigen::VectorXd & beta,const double nu, 
#                              const  Eigen::Map<Eigen::MatrixXd> & input,  
#                              const Eigen::MatrixXd & X,
#                              const std::string zero_mean,
#                              const Eigen::MatrixXd & output,
#                              const Eigen::MatrixXd & testing_input, 
#                              const Eigen::MatrixXd & X_testing, const Eigen::MatrixXd & L ,
#                              const Eigen::MatrixXd & LX,const Eigen::VectorXd & theta_hat,
#                                   double sigma2_hat,py::list rr0, py::list r0,
#                                   const Eigen::VectorXi & kernel_type,const Eigen::VectorXd & alpha,
#                                   const std::string method,const bool sample_data){

# construct_rgasp(const Eigen::VectorXd & beta,const double nu,
#                 const py::list& R0, const Eigen::MatrixXd & X,
#                 const  std:string zero_mean,const Eigen::MatrixXd & output,
#                 const Eigen::VectorXi & kernel_type,
#                 const Eigen::VectorXd & alpha){

# py::list pred_rgasp(const Eigen::VectorXd & beta,const double nu, const  Eigen::MatrixXd & input,  
#                     const Eigen::MatrixXd & X,const  std::string zero_mean, const Eigen::MatrixXd & output,
#                     const Eigen::MatrixXd & testing_input, const Eigen::MatrixXd & X_testing,
#                 const Eigen::MatrixXd & L , Eigen::MatrixXd & LX, Eigen::MatrixXd& theta_hat, 
#                 double sigma2_hat,double q_025, double q_975, List r0,const Eigen::VectorXi & kernel_type,
#                 const Eigen::VectorXd &alpha,const std::string method, const bool interval_data){



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


# ##########Borehole examples to test findInertInputs()

# sampler = qmc.LatinHypercube(d=8)
# sample_input = 10 * sampler.random(n=40)

# #input <- maximinLHS(n=40, k=8)  # maximin lhd sample
# # rescale the design to the domain of the Borehole function
# LB=np.array([0.05,100,63070,990,63.1,700,1120,9855])
# UB=np.array([0.15,50000,115600,1110,116,820,1680,12045])
# range_UL=UB-LB
# for i in range(8):
    
#     sample_input[:,i]=LB[i]+range_UL[i]*sample_input[:,i]

# num_obs=sample_input.shape[0]
# sample_output= np.zeros((num_obs,1))

# for i in range(num_obs):
#     sample_output[i,0]=borehole(sample_input[i,:])

# task = P_rgasp.create_task(sample_input, sample_output, lower_bound = False)  # optimization='nelder-mead'
# model = P_rgasp.train(task)
# #m<- rgasp(design = input, response = output, lower_bound=FALSE)
# P = findInertInputs(model)
