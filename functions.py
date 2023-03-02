#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 13:34:06 2023

@author: lihao
"""
import numpy as np
import scipy as sp
import cppimport


fcpp = cppimport.imp("my_own")

def higdon_1_data(s):
    """
    

    Parameters
    ----------
    # Implementation of the function in Higdon, D. (2002). 
    #  Space and space-time modeling using process convolutions. 
    #  In Quantitative methods for current environmental issues (pp. 37-56).
    #  Springer London.

    # First term: large scale variation
    # Second term: a fifth of the variation at four times the frequency

    Returns
    -------
    None.

    """
    pi = np.pi
    return np.sin(2*pi*s/10) + 0.2 * np.sin(2*pi*s/2.5)

def limetal_2_data(x):
    #############2 dimension example function from Lim et al. 2002
    res1 = 30 + 5*x[0]*np.sin(5*x[0])
    res2 = 4 + np.exp(-5*x[1])
    
    return ((res1*res2 - 100) / 6)
    



#############a 3 dimensional example function from Dette and Pepelyshev 2010
def dettepepel_3_data(x):

    res1 = 4 * (x[0] - 2 + 8*x[1] - 8*x[1]**2)**2
    res2 = (3 - 4*x[1])**2
    res3 = 16 * np.sqrt(x[2]+1) * (2*x[2]-1)**2
    
    return (res1 + res2 + res3)


# The Borehole Function from Worley (1987)
def borehole(x):
    
    rw = x[0]
    r  = x[1]
    Tu = x[2]
    Hu = x[3]
    Tl = x[4]
    Hl = x[5]
    L  = x[6]
    Kw = x[7]
      
    res1 = 2 * np.pi * Tu * (Hu-Hl)
    res2 = np.log(r/rw)
    res3 = 1 + (2*L*Tu / (res2*rw**2*Kw)) + (Tu / Tl)
      
    return res1/res2/res3

def environ_4_data(x, s=[0.5, 1, 1.5, 2, 2.5], t=np.arange(0.3, 60, 0.3)):
    

    M   = x[0]
    D   = x[1]
    L   = x[2]
    tau = x[3]
    
    ds = len(s)
    dt = len(t)
    dY = ds * dt
    Y = np.zeros((ds,dt))
   
    for ii in range(ds):
        si = s[ii]
        for jj in range(dt):
            tj = t[jj]
            term1a = M / np.sqrt(4*np.pi*D*tj)
            term1b = np.exp(-si**2 / (4*D*tj))
            term1 = term1a * term1b
            
            term2 = 0
            if tau < tj:
              term2a = M / np.sqrt(4*np.pi*D*(tj-tau))
              term2b = np.exp(-(si-L)**2 / (4*D*(tj-tau)))
              term2 = term2a * term2b
            
            
            C = term1 + term2
            Y[ii, jj] = np.sqrt(4*np.pi) * C
    
    Yrow = Y.T
    y = Yrow.reshape(1,-1)
   
    return y

def friedman_5_data(x):
    return 10 * np.sin(np.pi*x[0]*x[1]) + 20 * (x[2]-0.5)**2 + 10*x[3] + 5*x[4]

#### hold for now !!!!   # equarion 14 from jointly robust paper
# def findInertInputs(object,threshold=0.1){
#   P_hat=object@p*object@beta_hat*object@CL/sum(object@beta_hat*object@CL)
#   index_inert=which(P_hat<threshold)
  
#   print('The estimated normalized inverse range parameters are : {}'.format(P_hat)+'\n')
#   if(length(which(P_hat<0.1))>0){
#     cat('The inputs ', index_inert, 'are suspected to be inert inputs','\n')
#   }else{
#     cat('no input is suspected to be an inert input', '\n')
#   }
#   P_hat
# }


def neg_log_marginal_post_approx_ref(param, nugget, nugget_est,R0,X,zero_mean,output,CL,a,b,kernel_type,alpha):
    
   #####this has mean X, we should also include the case where X is not zero
   #####
   param = np.array(param).reshape(-1,1)
   lml=fcpp.log_marginal_lik(param,nugget,nugget_est,R0,X,zero_mean,output,kernel_type,alpha)
   lp=fcpp.log_approx_ref_prior(param,nugget,nugget_est,CL,a,b)
     
   return -(lml+lp)

def neg_log_profile_lik(param,nugget, nugget_est,R0,X,zero_mean,output,CL,a,b,kernel_type,alpha):
    param = np.array(param).reshape(-1,1)
    lpl=fcpp.log_profile_lik(param,nugget,nugget_est,R0,X,zero_mean,output,kernel_type,alpha)
    return -lpl
    



def neg_log_marginal_lik(param,nugget, nugget_est,R0,X,zero_mean,output,CL,a,b,kernel_type,alpha):
  #####this has mean X, we should also include the case where X is not zero
  #####
  param = np.array(param).reshape(-1,1)
  lml=fcpp.log_marginal_lik(param,nugget,nugget_est,R0,X,zero_mean,output,kernel_type,alpha)
  
  return -lml
  
def neg_log_marginal_post_approx_ref_ppgasp(param,nugget, nugget_est,R0,X,zero_mean,output,CL,a,b,kernel_type,alpha):
  #####this has mean X, we should also include the case where X is not zero
  #####
  lml=fcpp.log_marginal_lik_ppgasp(param,nugget,nugget_est,R0,X,zero_mean,output,kernel_type,alpha)
  lp=fcpp.log_approx_ref_prior(param,nugget,nugget_est,CL,a,b)
  #print(param)
  #print(-(lml+lp))
  
  return -(lml+lp)
  


def neg_log_profile_lik_ppgasp(param,nugget, nugget_est,R0,X,zero_mean,output,CL,a,b,kernel_type,alpha):
  #####this has mean X, we should also include the case where X is not zero
  #####
  lpl=fcpp.log_profile_lik_ppgasp(param,nugget,nugget_est,R0,X,zero_mean,output,kernel_type,alpha)
  #print(param)
  #print(-(lml+lp))
  
  return -lpl

def neg_log_marginal_lik_ppgasp(param,nugget, nugget_est,R0,X,zero_mean,output,CL,a,b,kernel_type,alpha):
  #####this has mean X, we should also include the case where X is not zero
  #####
  lml=fcpp.log_marginal_lik_ppgasp(param,nugget,nugget_est,R0,X,zero_mean,output,kernel_type,alpha);
  #print(param)
  #print(-(lml+lp))
  
  return -lml

def neg_log_marginal_post_ref(param,nugget, nugget_est,R0,X,zero_mean,output,prior_choice,kernel_type,alpha):
  
  param = np.array(param).reshape(-1,1)
  lmp=fcpp.log_ref_marginal_post(param,nugget,nugget_est,R0,X,zero_mean,output,kernel_type,alpha)
  
  
  if prior_choice=='ref_xi':
     if nugget_est==True:###note that in this case nu is also needed to be transformed have tail properties
      #-sum(param)-lmp  ###this will let nu be non zero so we need to be careful
       return -sum(param[0:(len(param)-1)])-lmp ###this might be fine when intercept is in the trend
     else: ###no nugget
      return -sum(param)-lmp
     
  elif prior_choice=='ref_gamma':
    if nugget_est==True:###note that in this case nu is also needed to be transformed have tail properties
      return -2*sum(param[0:(len(param)-2)])-lmp 
    else: ###no nugget
      return -2*sum(param)-lmp
    
  
###ppgasp
def neg_log_marginal_post_ref_ppgasp(param,nugget, nugget_est,R0,X,zero_mean,output,prior_choice,kernel_type,alpha):
  
  lmp=fcpp.log_ref_marginal_post_ppgasp(param,nugget,nugget_est,R0,X,zero_mean,output,kernel_type,alpha)
  
  
  if prior_choice=='ref_xi':
    if nugget_est==True:###note that in this case nu is also needed to be transformed have tail properties
      #-sum(param)-lmp  ###this will let nu be non zero so we need to be careful
      return -sum(param[0:(len(param)-1)])-lmp ###this might be fine when intercept is in the trend
    else: ###no nugget
      return -sum(param)-lmp
    
  elif prior_choice=='ref_gamma':
    if nugget.est==True:###note that in this case nu is also needed to be transformed have tail properties
      return -2*sum(param[0:(len(param)-2)])-lmp 
    else: ###no nugget
      return -2*sum(param)-lmp
    
def neg_log_marginal_post_approx_ref_deriv(param,nugget,nugget_est,R0,X,zero_mean,output,CL,a,b,kernel_type,alpha):
  param = np.array(param).reshape(-1,1)
  lml_dev=fcpp.log_marginal_lik_deriv(param,nugget,nugget_est,R0,X,zero_mean,output,kernel_type,alpha)
  lp_dev=fcpp.log_approx_ref_prior_deriv(param,nugget,nugget_est,CL,a,b)
  
  return -(lml_dev+lp_dev)*np.exp(param)
    



def neg_log_profile_lik_deriv(param,nugget,nugget_est,R0,X,zero_mean,output,CL,a,b,kernel_type,alpha):
  lpl_dev=fcpp.log_profile_lik_deriv(param,nugget,nugget_est,R0,X,zero_mean,output,kernel_type,alpha)

  return -(lpl_dev)*np.exp(param)
  


def neg_log_marginal_lik_deriv(param,nugget,nugget_est,R0,X,zero_mean,output,CL,a,b,kernel_type,alpha):
  lml_dev=fcpp.log_marginal_lik_deriv(param,nugget,nugget_est,R0,X,zero_mean,output,kernel_type,alpha)
  
  return -(lml_dev)*np.exp(param)
  

def neg_log_marginal_post_approx_ref_deriv_ppgasp(param,nugget,nugget_est,R0,X,zero_mean,output,CL,a,b,kernel_type,alpha):
  lml_dev=fcpp.log_marginal_lik_deriv_ppgasp(param,nugget,nugget_est,R0,X,zero_mean,output,kernel_type,alpha)
  lp_dev=fcpp.log_approx_ref_prior_deriv(param,nugget,nugget_est,CL,a,b)
  
  return -(lml_dev+lp_dev)*np.exp(param)
  



def neg_log_profile_lik_deriv_ppgasp(param,nugget,nugget_est,R0,X,zero_mean,output,CL,a,b,kernel_type,alpha):
  lpl_dev=fcpp.log_profile_lik_deriv_ppgasp(param,nugget,nugget_est,R0,X,zero_mean,output,kernel_type,alpha)

  return -(lpl_dev)*np.exp(param)
  


def neg_log_marginal_lik_deriv_ppgasp(param,nugget,nugget_est,R0,X,zero_mean,output,CL,a,b,kernel_type,alpha):
  lml_dev=fcpp.log_marginal_lik_deriv_ppgasp(param,nugget,nugget_est,R0,X,zero_mean,output,kernel_type,alpha)
  
  return -(lml_dev)*np.exp(param)
  
def euclidean_distance(m1,m2):
    res = fcpp.euclidean_distance(m1,m2)
    
    return res
    

#############this is a function to search the lower bounds for range parameters beta
#####need R0 in the function

def construct_rgasp(model_beta_hat, model_nugget, model_R0, model_X, model_zero_mean,
                            model_output,kernel_type_num,model_alpha):
    
    model_beta_hat = np.array(model_beta_hat).reshape(-1,1)
    return_list = fcpp.construct_rgasp(model_beta_hat, model_nugget, model_R0, model_X, model_zero_mean,
                                model_output,kernel_type_num,model_alpha)
    return return_list

def pred_rgasp(beta_hat,nugget,input,X,zero_mean,output,
                     testing_input,testing_trend,L,LX,theta_hat,
                     sigma2_hat,qt_025,qt_975,r0,kernel_type_num,alpha,method,interval_data):
    
    beta_hat = np.array(beta_hat).reshape(-1,1)
    res = fcpp.pred_rgasp(beta_hat,nugget,input,X,zero_mean,output,
                         testing_input,testing_trend,L,LX,theta_hat,
                         sigma2_hat,qt_025,qt_975,r0,kernel_type_num,alpha,method,interval_data)
    return res


def search_LB_prob(param, R0,COND_NUM_UB,p,kernel_type,alpha,nugget):
  num_obs= R0[0].shape[0]
  #dim(R0[[1]])[1]
  propose_prob=np.exp(param)/(np.exp(param)+1)
  LB=[]
  for i_LB in range(p):
      LB.append(np.log(-np.log(propose_prob)/(np.max(R0[i_LB]))))
  # for( i_LB in 1:p){
  #   LB=c(LB, log(-log(propose_prob)/(max(R0[[i_LB]]))))    ###LB is log beta
  # }
  
  R=fcpp.separable_multi_kernel(R0,np.exp(LB),kernel_type,alpha)  
  
  # if(!isotropic){
  #   R=separable_multi_kernel(R0,exp(LB),kernel_type,alpha)  
  # }else{
  #   R=pow_exp_funct(R0[[1]],exp(LB),1)  
  # }
  #R=as.matrix(R)
  #R=R+nugget*diag(num_obs)
  R[np.diag_indices_from(R)] += nugget
  
  kappa_R = np.linalg.cond(R)
  return (kappa_R-COND_NUM_UB)**2
  ##one might change it to
  ##(kappa(R,exact=T)-COND_NUM_UB)^2
  
def leave_one_out_rgasp(object_class):
  R_tilde = object_class.L @ np.tranpose(object_class.L)+object_class.nugget
  sigma_2 = np.repeat(0,object_class.num_obs)
  mean = np.repeat(0,object_class.num_obs)
  if object_class.zero_mean == "Yes":
    for i in range(object_class.num_obs):

        #1:object_class@num_obs){
        
        r_sub = np.delete(R_tilde,i,0)[:,i] # R_tilde[-i,i]
        #L_sub = np.linalg.cholesky(np.delete(np.delete(R_tilde,i,0),i,1))
            #R_tilde[-i,-i])
            
        L, lower = sp.linalg.cho_factor(
            np.delete(np.delete(R_tilde,i,0),i,1), overwrite_a=True, check_finite=False
        )
        r_sub_t_R_sub_inv  = sp.linalg.cho_solve(
            (L, lower), r_sub, overwrite_b=True, check_finite=False
        )
        
        #r_sub_t_R_sub_inv=t(backsolve(t(L_sub),forwardsolve(L_sub,r_sub)))
        
        R_sub_inv_y = sp.linalg.cho_solve(
            (L, lower), np.delete(object_class.output,i), overwrite_b=True, check_finite=False
        )
        #R_sub_inv_y=backsolve(t(L_sub),forwardsolve(L_sub,object_class@output[-i] ))
        mean[i]=r_sub_t_R_sub_inv @ np.delete(object_class.output,i)
        sigma_2_hat=np.delete(object_class.output,i) @ R_sub_inv_y/(object_class.num_obs-1)
        sigma_2[i]=sigma_2_hat*(R_tilde[i,i]-r_sub_t_R_sub_inv @ r_sub)
    
  else:
    for i in range(object_class.num_obs):
        r_sub = np.delete(R_tilde,i,0)[:,i]
        L, lower = sp.linalg.cho_factor(
            np.delete(np.delete(R_tilde,i,0),i,1), overwrite_a=True, check_finite=False
        )
        r_sub_t_R_sub_inv  = sp.linalg.cho_solve(
            (L, lower), r_sub, overwrite_b=True, check_finite=False
        )
        
        R_inv_X = sp.linalg.cho_solve(
            (L, lower), np.delete(object_class.X,i,0), overwrite_b=True, check_finite=False
        )
        
        L_x, lower_x = sp.linalg.cho_factor(
            np.tranpose(np.delete(object_class.X,i,0)) @ R_inv_X, overwrite_a=True, check_finite=False
        )
        theta_hat  = sp.linalg.cho_solve(
            (L_x, lower_x), np.tranpose(R_inv_X)@np.delete(object_class.output,i) , overwrite_b=True, check_finite=False
        )

        #r_sub=R_tilde[-i,i]
        #L_sub=t(chol(R_tilde[-i,-i]))
        #r_sub_t_R_sub_inv=t(backsolve(t(L_sub),forwardsolve(L_sub,r_sub)))
        
        #R_inv_X=backsolve(t(L_sub),forwardsolve(L_sub,(object_class@X[-i,]) ))
  
          
        #L_X=t(chol(t(object_class@X[-i,])%*%R_inv_X))
        #theta_hat=backsolve(t(L_X),forwardsolve(L_X,t(R_inv_X)%*%object_class@output[-i]))
  
        tilde_output = np.delete(object_class.output,i)- np.delete(object_class.X,i,0)@theta_hat
        mean[i]=object_class.X[i,:] @ theta_hat+r_sub_t_R_sub_inv @ tilde_output
      
      
        if (object_class.method=='post_mode') or (object_class.method=='mmle'):
            
            sigma2_hat = np.tranpose(tilde_output) @ sp.linalg.cho_solve(
                (L, lower), tilde_output, overwrite_b=True, check_finite=False
            )/(object_class.num_obs-1-object_class.q)
            #backsolve(t(L_sub),forwardsolve(L_sub,tilde_output ))/(object_class@num_obs-1-object_class@q)
            
            c_star=(R_tilde[i,i]-r_sub_t_R_sub_inv @ r_sub)
            
      
            h_hat=object_class.X[i,:]-np.tranpose(np.delete(object_class.X,i,0)) @ np.tranpose(r_sub_t_R_sub_inv)
            c_star_star= c_star+ np.tranpose(h_hat) @ sp.linalg.cho_solve(
                (L_x, lower_x), h_hat, overwrite_b=True, check_finite=False
            )
            
            #backsolve(t(L_X),forwardsolve(L_X,h_hat))
            sigma_2[i] = sigma2_hat*(c_star_star)
        elif object_class@method=='mle':
          sigma2_hat=np.tranpose(tilde_output) @ sp.linalg.cho_solve(
              (L, lower), tilde_output, overwrite_b=True, check_finite=False
          )/(object_class.num_obs-1)
          #t(tilde_output)%*%backsolve(t(L_sub),forwardsolve(L_sub,tilde_output ))/(object_class@num_obs-1)
          
          c_star=(R_tilde[i,i]-r_sub_t_R_sub_inv @ r_sub)
          
          sigma_2[i]=sigma2_hat*(c_star)
        
      #sigma_2[i]=object_class@sigma2_hat*(c_star)
      
    
  
  
  return {'mean':mean,'sd':np.sqrt(sigma_2)}

  
  #plot((output-mean)/sqrt(sigma_2))


  

    
# a1 = np.column_stack((,b))np
# Eigs = np.linalg.eigvals(a1)
# cond = np.max(Eigs)/np.min(Eigs) 

# import scipy 
# A = a1
# norm_A = scipy.sparse.linalg.norm(A)
# norm_invA = scipy.sparse.linalg.norm(scipy.sparse.linalg.inv(A))
# cond = norm_A*norm_invA

# a = a1
# np.max(np.linalg.qr(a))*np.min(np.linalg.qr(np.linalg.inv(a)))





A = np.array([[0.1,0.2],
              [0.2,0.1]])
X = np.array([[0.1,0.2],
              [0.3,0.4]])
L = np.array([[0.1,0],
              [0.2,0.1]])

#log_approx_ref_prior(np.array([0.1,0.1]),0.1, True,np.array([0.1,0.1]),0.1,0.1)

#print("friedman_5_data",friedman_5_data([1,2,3,4,5]))  
print(neg_log_marginal_lik_deriv_ppgasp(np.array([0.1,0.1]),0.1, True,[A,A],A,"Yes",X, np.array([0.1,0.1]),0.1,0.2, np.array([1,2]),np.array([0.1,0.1])))

#np.array([0.1,0.1]),0.1,0.2
#print(neg_log_marginal_post_ref_ppgasp(np.array([0.1,0.1]),0.1, True,[A,A],A,"Yes",X, 'ref_xi', np.array([1,2]),np.array([0.1,0.1])))

#print(neg_log_marginal_post_approx_ref(np.array([0.1,0.1]),0.1, True,[A,A],A,"Yes",A,np.array([0.1,0.1]),0.1,0.1,np.array([1,2]),np.array([0.1,0.1])))
# print("higdon_1_data",higdon_1_data(5))  
# print("limetal_2_data",limetal_2_data([1,2]))  
# print("borehole",borehole([1,2,3,4,5,6,7,8]))          
# print("dettepepel_3_data",dettepepel_3_data([1,2,3]))                                  
#print("environ_4_data",environ_4_data([1,2,3,4]))                 
 

