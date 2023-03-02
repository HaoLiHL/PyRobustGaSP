#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:40:25 2023

@author: lihao
"""

##########################################################################
## rgasp fit function
## 
## Robust GaSP Package
##
## This software is distributed under the terms of the GNU GENERAL
## PUBLIC LICENSE Version 2, April 2013.
##
## Copyright (C) 2015-present Mengyang Gu, Jesus Palomo, James O. Berger
##							  
##    
##########################################################################
#library(nloptr)    ####I need to have this nloptr, or I can use C++ nlopt, this one is faster than optim()
#requireNamespace("nloptr")

# design is a input matrix n times p 
import sys 
import numpy as np
import scipy as sp
from scipy.optimize import minimize,fmin_l_bfgs_b
import scipy.stats    

from functions import *

class rgasp(object):
    def __init__(self):
        
        return
        
        
    def create_task(self,design,response,
                    zero_mean="No",
                    nugget=0,
                    nugget_est=False,
                    range_par=None,
                    method='post_mode',
                    prior_choice='ref_approx',
                    
                    kernel_type=['matern_5_2'],
                    isotropic=False,
                    R0=None,
                    optimization='lbfgs',
                    #alpha=np.repeat(1.9,design.shape[1]),
                    lower_bound=True,
                    #max_eval=max(30,20+5*design.shape[1]),
                    initial_values=None,
                    num_initial_values=2):

        # self.design = design
        # self.response = response
        # self.nugget = nugget
        # self.nugget_est = nugget_est
        # self.range_par = range_par
        # self.method = method
        # self.prior_choice = prior_choice
        

        # self.trend=np.repeat(1,len(self.response)).reshape(-1,1),      #matrix(1,length(response),1),
        # self.a=0.2,
        # self.b=1/(len(self.response))**(1/self.design.shape[1])*(a+self.design.shape[1]),
        # self.kernel_type = kernel_type
        # self.isotropic = isotropic
        # self.R0 = R0
        # self.optimization = optimization
        # self.lower_bound = lower_bound
        # self.initial_values = initial_values
        # self.num_initial_values = num_initial_values
        # #kernel_type='matern_5_2',
        # # isotropic=False,
        # # R0=None,
        # # optimization='lbfgs',
        # self.alpha=np.repeat(1.9,self.design.shape[1]),
        # #lower_bound=True,
        # self.max_eval=max(30,20+5*design.shape[1]),
        
        
        trend=np.repeat(1.0,len(response)).reshape(-1,1)
        max_eval=max(30,20+5*design.shape[1])
        alpha=np.repeat(1.9,design.shape[1])
        a=0.2
        b=1/(len(response))**(1/design.shape[1])*(a+design.shape[1])
        task = {

            'design': design,
            'response': response,
            'trend': trend,
            'zero_mean': zero_mean,
            'nugget': nugget,
            'nugget_est': nugget_est,
            'range_par': range_par,
            'method': method,
            'prior_choice': prior_choice,
            'a': a,
            'b': b,
            'kernel_type': kernel_type,
            'isotropic': isotropic,
            'R0': R0,
            'optimization': optimization,
            'alpha': alpha,
            'lower_bound': lower_bound,
            'max_eval': max_eval,
            'initial_values': initial_values,
            'num_initial_values': num_initial_values,           
        }
        #initial_values=None,
        #num_initial_values=2):
        
        if zero_mean=="Yes":
            trend=np.repeat(0.0,len(response)).reshape(-1,1)
            
        if (type(nugget_est)!=bool ):   # (!is.logical(nugget.est) && length(nugget.est) != 1)
            sys.exit('nugget.est should be boolean (either T or F) \n')
            #and len(nugget_est)!= 1)

        if (nugget!=0 and  nugget_est==True):
          sys.exit("one cannot fix and estimate the nugget at the same time \n")  
        
      
        if (range_par!=None): 
          if len(range_par)!=design.shape[1]:
            sys.exit("range.par should either be fixed or estimated.")    
          
          if nugget_est:
            sys.exit("We do not support fixing range parameters while estimating the nugget.")      
          
            
        if (not (isinstance(nugget, (int, float, complex)) and not isinstance(nugget, bool))):
          sys.exit("nugget should be a numerical value \n")  
        
        
        if ( (optimization!='lbfgs') and (optimization!='nelder-mead') and (optimization!='brent')):
          sys.exit("optimization should be 'lbfgs' or 'nelder-mead' or 'brent' \n")  
          
          
        if (type(nugget_est)!=bool):
          sys.exit("isotropic should be either true or false \n")  
        
          
        
            
        return task
    
    def train(self,task):
        
        #####Only numeric inputs are allowed
        design=task['design']
        n = design.shape[0]
        p_x = design.shape[1]
        
        
        
        model_input = design # <- matrix(as.numeric(design), dim(design)[1],dim(design)[2])
        
        # print(model@input)
        model_output = task['response'].reshape(-1,1)
        # print(model@output)
        #p_x <- dim(model_input)[2]
        kernel_type= task['kernel_type']

        model_isotropic=task['isotropic']
        
        if not model_isotropic:
          model_alpha = task['alpha']
        else:
          model_alpha=task['alpha'][0]
        model_alpha = model_alpha.reshape(-1,1)
        
        if(model_isotropic):
          model_p = 1
        else:
          model_p=p_x
        
        if (task['optimization']=='brent' and model_p!=1):
          sys.exit('The Brent optimization method can only work for optimizing one parameter \n')
        
        
        model_num_obs = n
        
        ## Checking the dimensions between the different inputs
        if (model_num_obs != model_output.shape[0]):
          sys.exit("The dimensions of the design matrix and the response do not match. \n")
        
        #  stop("The dimension of the training points and simulator values are different. \n")
        
        ###add method 
        method = task['method']
        if ( (method!='post_mode') and (method!='mle') and (method!='mmle') ):
          sys.exit("The method should be post_mode or mle or mmle. \n")
        
        
        model_method=method
        
        
        if (not model_isotropic):
            
            if (len(kernel_type)==1):
                
                model_kernel_type=np.repeat(kernel_type, model_p)
        
            

            elif (len(kernel_type)!=model_p):
                
                
                sys.exit("Please specify the correct number of kernels. \n")
            else:
                model_kernel_type=kernel_type
           
        else:
            model_kernel_type=kernel_type
        
         ##model@kernel_type <-kernel_type
         
         ##change kernel type to integer to pass to C++ code
         ##1 is power exponent, 2 is matern with roughness 3/2, and 3 is matern with roughenss parameter 5/2
        kernel_type_num=np.repeat(0,model_p)#rep(0,  model@p)
        for i_p in range(model_p):
          if (model_kernel_type[i_p]=="matern_5_2"):
              
              kernel_type_num[i_p]=3
          elif (model_kernel_type[i_p]=="matern_3_2"):
              kernel_type_num[i_p]=2
          elif (model_kernel_type[i_p]=="pow_exp"):
              kernel_type_num[i_p]=1
          elif (model_kernel_type[i_p]=="periodic_gauss"): ##this is periodic folding on Gaussian kernel
              kernel_type_num[i_p]=4
          elif (model_kernel_type[i_p]=="periodic_exp"):   ##this is periodic folding on Exponential kernel
              kernel_type_num[i_p]=5
        
        kernel_type_num = kernel_type_num.reshape(-1,1)
        #####I now build the gasp emulator here
        ##default setting constant mean basis. Of course We should let user to specify it
        #model@X = matrix(1,model@num_obs,1)   ####constant mean
        model_X=task['trend']              ###If the trend is specified, use it. If not, use a constant mean. 
        model_zero_mean=task['zero_mean']
        #######number of regressor
        if (model_zero_mean=="Yes"):
            
            
            model_q=0
        else:
            model_q = model_X.shape[1] # NOTE THIS IS ALWAYS 1 SINCE YOU DEFINE IT THAT WAY ABOVE
        
        ####################correlation matrix
        
        model_nugget_est = task['nugget_est']
        R0 = task['R0']
        
        if(R0 ==None):
            ##no given value
            if (not model_isotropic):
                
                model_R0 = []#as.list(1:model@p)
                for i in range(model_p):#1:model_p):
                  model_R0.append( np.abs(model_input[:,i][:,None] - model_input[:,i]) ) 
                  #= as.matrix(abs(outer(model@input[,i], model@input[,i], "-")))
                
            else:
              model_R0 = []
              if (p_x<model_num_obs):
                  
                  R0_here=0
                  for i in range(p_x):
                      
                    R0_here=R0_here+np.abs(model_input[:,i][:,None] - model_input[:,i])**2
                  
                  model_R0.append(np.sqrt(R0_here))
              else:
                  model_R0.append(euclidean_distance(model_input,model_input))
                
            
        elif (type(R0) == np.ndarray):
            model_R0=[R0]
            
            
         
        elif (type(R0) ==list):
            model_R0=R0
        else:
            sys.exit("R0 should be either a matrix or a list \n")
        
        
        ##check for R0
        if(len(model_R0)!=model_p):
            sys.exit("the number of R0 matrices should be the same as the number of range parameters in the kernel \n")
        
        if ( (model_R0[0].shape[0]!=model_num_obs) or (model_R0[0].shape[1]!=model_num_obs)):
            
            sys.exit("the dimension of R0 matrices should match the number of observations \n")
        
        # line 197   
     
        ###########calculating lower bound for beta
        model_CL = np.repeat(0.0,model_p)# rep(0,model@p)    ###CL is also used in the prior so I make it a model parameter
        
        if (not model_isotropic):
            
            
            for i_cl in range(model_p):
                print(type((np.max(model_input[:,i_cl])-np.min(model_input[:,i_cl]))/model_num_obs**(1/model_p))
          )
                
                model_CL[i_cl] = ((np.max(model_input[:,i_cl])-np.min(model_input[:,i_cl]))/model_num_obs**(1/model_p))
          
        else:
            model_CL[0]=np.max(model_R0[0])/model_num_obs
            
        range_par = task['range_par']
        lower_bound = task['lower_bound']
        nugget = task['nugget']
        initial_values = task['initial_values']
        num_initial_values = task['num_initial_values']
        a = task['a']
        b = task['b']
        optimization = task['optimization']
        prior_choice = task['prior_choice']
        nugget_est = task['nugget_est']
        
        
        if (range_par ==None):
            
            COND_NUM_UB = 10**(16)  ###maximum condition number, this might be a little too large
            
            if (lower_bound==True):
                
                bnds = ((-5, 12),)
                LB_all = minimize(search_LB_prob,[0],bounds = bnds,args= (model_R0,COND_NUM_UB,model_p,kernel_type_num,model_alpha,nugget,))
                # optimize(search_LB_prob, interval=c(-5,12), maximum = FALSE, R0=model@R0,COND_NUM_UB= COND_NUM_UB,
                #                   p=model@p,kernel_type=kernel_type_num,alpha=model@alpha,nugget=nugget) ###find a lower bound for parameter beta
                LB_all_minimum = LB_all.x[0]
                LB_prob = np.exp(LB_all_minimum)/(np.exp(LB_all_minimum)+1)
                
                LB = []
                
                for i_LB in range(model_p):
                    
                    #LB = c(LB, np.log(-log(LB_prob)/(np.max(model_R0[i_LB]))))    ###LB is lower bound for log beta, may consider to have it related to p
                    LB.append(np.log(-np.log(LB_prob)/(np.max(model_R0[i_LB]))))
            else:
                
                ##give some empirical bound that passes the initial values if lower bound is F
                ##could can change the first initial value if not search the bound
                LB = []
                
                for i_LB in range(model_p):
                    LB.append(-np.log(0.1)/ ((np.max(model_input[:,i_LB])-np.min(model_input[:,i_LB]))*model_p) )
                    #LB = c(LB, -log(0.1)/((max(model@input[,i_LB])-min(model@input[,i_LB]))*model@p)) 
                    # Line 241
            
            
            if lower_bound==True:
                if model_nugget_est:
                  model_LB = LB[:] + [-np.inf]
                  print(model_LB)
                else:
                  model_LB= LB[:]
                  
                #model_LB = np.array(model_LB)
              
            else:
                if model_nugget_est:
                    
                    
                    model_LB=np.repeat(-np.inf,model_p+1)  
                else:
                    model_LB=np.repeat(-np.inf,model_p)
            
            print('The upper bounds of the range parameters are',1/np.exp(model_LB),'\n')
            
            if (initial_values==None):

                beta_initial=np.zeros((num_initial_values,model_p))
                eta_initial=np.repeat(0.0,num_initial_values)
                beta_initial[0,:]=50*np.exp(LB) #####one start
                eta_initial[0]=0.0001
                if(num_initial_values>1):
                    
                    beta_initial[1,]=(a+model_p)/(model_p*model_CL*b)/2  ###half the prior value
                    eta_initial[1]=0.0002
                
                if(num_initial_values>2):
                    
                    for i_ini in range(2,num_initial_values):#3:num_initial_values){
                        #set.seed(i_ini)
                        np.random.seed(i_ini)
                        beta_initial[i_ini,:]=10**3*np.random.uniform(size = model_p)/model_CL
                        eta_initial[i_ini]=10**(-3)*np.random.uniform()
                      
                
                initial_values=np.hstack((np.log(beta_initial),np.log(eta_initial).reshape(-1,1)))
                
            if (method=='post_mode'):
                
                object_funct_name='marginal posterior'
            elif (method=='mle'):
                
                object_funct_name='profile likelihood'
            else:
                object_funct_name='marginal likelihood'
            # line 289
            
            model_log_post=-np.inf
            
            bounds = tuple((val,None) for val in model_LB)
            max_eval = task['max_eval']
            
            model_beta_hat = None
            
            if(optimization=='lbfgs'):
                
                for i_ini in range(num_initial_values):
                    
                    if (model_nugget_est):
                        
                        ini_value=initial_values[i_ini,:]
                    else:
                        
                        ini_value=initial_values[i_ini,0:model_p]
                      ###without the nugget
                    print('The initial values of range parameters are', 1/np.exp(ini_value[0:model_p]),'\n')
                    print('Start of the optimization ', i_ini,' : \n')
                    
                    try:
                        
                        if (method=='post_mode'):
                            if (prior_choice=='ref_approx'):
                                
                                ####this one can be with nugget or without the nugget
                              #  if (requireNamespace("lbfgs", quietly = TRUE)) {
                                tt_all = minimize(neg_log_marginal_post_approx_ref, ini_value,jac=neg_log_marginal_post_approx_ref_deriv,method = 'L-BFGS-B',
                                                                   args = (nugget,model_nugget_est,model_R0,model_X,model_zero_mean,model_output, model_CL, a,b,
                                                                   kernel_type_num,model_alpha,),bounds = bounds,  options = {'maxiter': max_eval})
                                #print(tt_all.x)
                                # tt_all <- try(nloptr::lbfgs(ini_value, neg_log_marginal_post_approx_ref, 
                                #                             neg_log_marginal_post_approx_ref_deriv,nugget=nugget, nugget.est=model@nugget.est, 
                                #                             R0=model@R0,X=model@X, zero_mean=model@zero_mean,output=model@output, CL=model@CL, a=a,b=b,
                                #                             kernel_type=kernel_type_num,alpha=model@alpha,lower=model@LB,
                                #                             nl.info = FALSE, control = list(maxeval=max_eval)),TRUE)
                                #   }
                            elif (prior_choice=='ref_xi' or prior_choice=='ref_gamma'):
                                ####this needs to be revised
                              #  if (requireNamespace("lbfgs", quietly = TRUE)) {
                              
                                tt_all = minimize(neg_log_marginal_post_ref, ini_value,method = 'L-BFGS-B',
                                                                   args = (nugget,model_nugget_est,model_R0,model_X,model_zero_mean,model_output, prior_choice,
                                                                   kernel_type_num,model_alpha,),bounds = bounds, options = {'maxiter': max_eval})
                             
                                # tt_all = optimize(neg_log_marginal_post_ref, ini_value,approx_grad = True,
                                #                                    args = (nugget,model_nugget_est,model_R0,model_X,model_zero_mean,model_output, prior_choice,
                                #                                    kernel_type_num,model_alpha,),bounds = bounds, m = max_eval)
                              
                                # tt_all <- try(nloptr::lbfgs(ini_value, neg_log_marginal_post_ref, 
                                #                             nugget=nugget, nugget.est=nugget.est, R0=model@R0,
                                #                             X=model@X, zero_mean=model@zero_mean,output=model@output, prior_choice=prior_choice, kernel_type=kernel_type_num,
                                #                             alpha=model@alpha,lower=model@LB,nl.info = FALSE, control = list(maxeval=max_eval)),TRUE)
                                # # }
                            #line 320
                        elif (method=='mle'):
                            
                            tt_all = minimize(neg_log_profile_lik, ini_value,jac = neg_log_profile_lik_deriv,method = 'L-BFGS-B',
                                                               args = (nugget,model_nugget_est,model_R0,model_X,model_zero_mean,model_output, model_CL,a,b,
                                                               kernel_type_num,model_alpha,),bounds = bounds, options = {'maxiter': max_eval})
                         
                            
                            # tt_all <- try(nloptr::lbfgs(ini_value, neg_log_profile_lik, 
                            #                           neg_log_profile_lik_deriv,nugget=nugget, nugget.est=model@nugget.est, 
                            #                           R0=model@R0,X=model@X, zero_mean=model@zero_mean,output=model@output, CL=model@CL, a=a,b=b,
                            #                           kernel_type=kernel_type_num,alpha=model@alpha,lower=model@LB,
                            #                           nl.info = FALSE, control = list(maxeval=max_eval)),TRUE)
                        elif (method == 'mmle'):
                            # tt_all = sp.optimize.fmin_l_bfgs_b(neg_log_marginal_lik, ini_value,fprime = neg_log_marginal_lik_deriv,
                            #                                    args = (nugget,model_nugget_est,model_R0,model_X,model_zero_mean,model_output, model_CL,a,b,
                            #                                    kernel_type_num,model_alpha,),bounds = bounds, m = max_eval)
                            tt_all = minimize(neg_log_marginal_lik, ini_value,jac = neg_log_marginal_lik_deriv,method = 'L-BFGS-B',
                                                               args = (nugget,model_nugget_est,model_R0,model_X,model_zero_mean,model_output, model_CL,a,b,
                                                               kernel_type_num,model_alpha,),bounds = bounds, options = {'maxiter': max_eval})
                    except:
                        # if an error detected from above optimization method
                        print('An error detected from above optimization method')
                    
                    
                    if (model_nugget_est==False):   
                        nugget_par=nugget
                    else:
                        nugget_par=np.exp(tt_all.x)[model_p]
                    
                    print('The number of iterations is ', tt_all.nit,'\n')
                    print('The value of the ', object_funct_name, '\n')
                    print(' function is ', -tt_all.fun,'\n')
                    print('Optimized range parameters are', 1/np.exp(tt_all.x)[0:model_p],'\n')
                    print('Optimized nugget parameter is', nugget_par,'\n')
                    print('Convergence: ', tt_all.success,'\n' )
                    
                    if ( (-tt_all.fun)>=model_log_post) or (model_beta_hat == None):
                        
                        log_lik=-tt_all.fun
                        model_log_post=-tt_all.fun
                        model_nugget=nugget
                        if (nugget_est):
                            
                            model_beta_hat = np.exp(tt_all.x)[0:model_p]
                            model_nugget=np.exp(tt_all.x)[model_p]
                        else:
                            
                            model_beta_hat = np.exp(tt_all.x)
                          #  model@nugget=0;
                        
                    
                        
                     
                    
                    
                        # line 336
                    
            
            elif optimization=='nelder-mead':
                
                for i_ini in range(num_initial_values):
                    
                    if (model_nugget_est):
                        
                        ini_value=initial_values[i_ini,:]
                    else:
                        
                        ini_value=initial_values[i_ini,0:model_p]
                      ###without the nugget
                    print('The initial values of range parameters are', 1/np.exp(ini_value[0:model_p]),'\n')
                    print('Start of the optimization ', i_ini,' : \n')
                    
                    if (method=='post_mode'):
                        if (prior_choice=='ref_approx'):
                            
                            ####this one can be with nugget or without the nugget
                          #  if (requireNamespace("lbfgs", quietly = TRUE)) {
                            tt_all = minimize(neg_log_marginal_post_approx_ref, ini_value,method = 'Nelder-Mead',
                                                               args = (nugget,model_nugget_est,model_R0,model_X,model_zero_mean,model_output, model_CL, a,b,
                                                               kernel_type_num,model_alpha,))
                        elif (prior_choice=='ref_xi' or prior_choice=='ref_gamma'):
                            ####this needs to be revised
                          #  if (requireNamespace("lbfgs", quietly = TRUE)) {
                          
                            tt_all = minimize(neg_log_marginal_post_ref, ini_value,method = 'Nelder-Mead',
                                                               args = (nugget,model_nugget_est,model_R0,model_X,model_zero_mean,model_output, prior_choice,
                                                               kernel_type_num,model_alpha,))
                    elif (method=='mle'):
                        tt_all = minimize(neg_log_profile_lik, ini_value,method = 'Nelder-Mead',
                                                           args = (nugget,model_nugget_est,model_R0,model_X,model_zero_mean,model_output, model_CL,a,b,
                                                           kernel_type_num,model_alpha,))
                    elif (method == 'mmle'):
                        tt_all = minimize(neg_log_marginal_lik, ini_value,jac = neg_log_marginal_lik_deriv,method = 'Nelder-Mead',
                                                           args = (nugget,model_nugget_est,model_R0,model_X,model_zero_mean,model_output, model_CL,a,b,
                                                           kernel_type_num,model_alpha,))
                    
                    if (model_nugget_est==False):   
                        nugget_par=nugget
                    else:
                        nugget_par=np.exp(tt_all.x)[model_p]   
                    print('The number of iterations is ', tt_all.nit,'\n')
                    print('The value of the ', object_funct_name, '\n')
                    print(' function is ', -tt_all.fun,'\n')
                    print('Optimized range parameters are', 1/np.exp(tt_all.x)[0:model_p],'\n')
                    print('Optimized nugget parameter is', nugget_par,'\n')
                    print('Convergence: ', tt_all.success,'\n' )
                    
                    if ( (-tt_all.fun)>model_log_post):
                        
                        log_lik=-tt_all.fun
                        model_log_post=-tt_all.fun
                        model_nugget=nugget
                        if (nugget_est):
                            
                            model_beta_hat = np.exp(tt_all.x)[0:model_p]
                            model_nugget=np.exp(tt_all.x)[model_p]
                        else:
                            
                            model_beta_hat = np.exp(tt_all.x)
                        
                
            elif optimization=='brent':
                
                
                
                if (method=='post_mode'):
                    
                    UB=max(20, 1/np.log(np.max(model_R0[0])))
                    
                    bounds = tuple((val,UB) for val in LB)
                    #max_eval = task['max_eval']
                    
                    if (prior_choice=='ref_approx'):
                        ####this one can be with nugget or without the nugget
                        
                        tt_all = sp.optimize.minimize_scalar(neg_log_marginal_post_approx_ref,method = 'Brent',
                                                           args = (nugget,model_nugget_est,model_R0,model_X,model_zero_mean,model_output, model_CL, a,b,
                                                           kernel_type_num,model_alpha,), bounds = bounds)
                        
                        # tt_all <- try(optimize(neg_log_marginal_post_approx_ref,nugget=nugget, nugget.est=model@nugget.est, 
                        #                        R0=model@R0,X=model@X, zero_mean=model@zero_mean,output=model@output, CL=model@CL, a=a,b=b,
                        #                        kernel_type=kernel_type_num,alpha=model@alpha,lower=LB,upper=UB),TRUE)
                        
                    elif (prior_choice=='ref_xi' or prior_choice=='ref_gamma'):
                        tt_all = sp.optimize.minimize_scalar(neg_log_marginal_post_ref,method = 'Brent',
                                                           args = (nugget,model_nugget_est,model_R0,model_X,model_zero_mean,model_output, prior_choice,
                                                                   kernel_type_num,model_alpha,), bounds = bounds)
                        
                elif (method=='mle'):
                    UB=max(20, 1/np.log(np.max(model_R0[0])))
                    
                    bounds = tuple((val,UB) for val in LB)
                    tt_all = sp.optimize.minimize_scalar(neg_log_profile_lik,method = 'Brent',
                                                       args = (nugget,model_nugget_est,model_R0,model_X,model_zero_mean,model_output, model_CL,a,b,
                                                       kernel_type_num,model_alpha,),bounds = bounds)
                elif (method == 'mmle'):
                    UB=max(20, 1/np.log(np.max(model_R0[0])))
                    
                    bounds = tuple((val,UB) for val in LB)
                    tt_all = sp.optimize.minimize_scalar(neg_log_marginal_lik,method = 'Brent',
                                                       args = (nugget,model_nugget_est,model_R0,model_X,model_zero_mean,model_output, model_CL,a,b,
                                                       kernel_type_num,model_alpha,),bounds = bounds)
                
                print('The number of iterations is ', tt_all.nit,'\n')
                print('The value of the ', object_funct_name, '\n')
                print(' function is ', -tt_all.fun,'\n')
                print('Optimized range parameters are', 1/np.exp(tt_all.x),'\n')
                print('Convergence: ', tt_all.success,'\n' )
                
                
                model_log_post=-tt_all.fun
                model_nugget=nugget
                model_beta_hat = np.exp(tt_all.x)
                       
        
        else:
            #this is the case where the range parameters and the nugget are all fixed
            model_LB=np.repeat(-np.inf,model_p)
            model_beta_hat=1/range_par
            model_nugget=nugget
            
        
        list_return = construct_rgasp(model_beta_hat, model_nugget, model_R0, model_X, model_zero_mean,
                                    model_output,kernel_type_num,model_alpha)
        
        
        model_L=list_return[0]
        model_LX=list_return[1]
        model_theta_hat=list_return[2]
        model_sigma2_hat = None
        if ((method=='post_mode') or (method=='mmle') ):
            
            
            model_sigma2_hat=list_return[3]
        elif (method=='mle'):
            
          #if(model@q>0){
            model_sigma2_hat=list_return[3]*(model_num_obs-model_q)/model_num_obs
          #}
        
        #return(model)
        
        model = {'beta_hat':model_beta_hat,
                        'nugget':model_nugget,
                        'R0':model_R0,
                        'X':model_X,
                        'zero_mean':model_zero_mean,
                        'output':model_output,
                        'kernel_type_num':kernel_type_num,
                        'alpha':model_alpha,
                        'L':model_L,
                        'LX':model_LX,
                        'theta_hat':model_theta_hat,
                        'sigma2_hat':model_sigma2_hat,
                        'isotropic':model_isotropic,
                        'p':model_p,
                        'num_obs': model_num_obs,
                        'input': model_input,
                        'method': model_method,
                        'q': model_q,
                        'nugget_est': model_nugget_est,
                        'kernel_type':kernel_type
                        
                        
                        
                        
                       
                        }    
        
        print('Mean parameters: ',model['theta_hat'],'\n')
        print('Variance parameter: ', model['sigma2_hat'],'\n')
        print('Range parameters: ', 1/model['beta_hat'],'\n')
        print('Noise parameter: ', model['sigma2_hat']*model['nugget'],'\n')
        return model
    
    
    
    def predict_rgasp(self,model, 
                      testing_input, 
                      testing_trend= None,
                      r0= None,
                     interval_data=True,
                     outasS3 = True):
        
        if testing_trend == None:
            testing_trend = np.repeat(1.0,testing_input.shape[0]).reshape(-1,1)
        
        testing_input=np.array(testing_input)
        
        if(model['zero_mean']=="Yes"):
            
            testing_trend=np.repeat(0.0,testing_input.shape[0])
        else:
            if testing_trend.shape[1]!=model['X'].shape[1]:
                sys.exit("The dimensions of the design trend matrix and testing trend matrix do not match. \n")
                
        if ( testing_input.shape[1]!=model['input'].shape[1]):  
            sys.exit("The dimensions of the design matrix and testing inputs matrix do not match. \n")
        
        num_testing_input = testing_input.shape[0]
        #X_testing = matrix(1,num_testing_input,1) ###testing trend

        testing_input=np.array(testing_input)

        ##form the r0 matrix
        p_x = model['input'].shape[1]
        
        if (r0 == None):
            
            if( not model['isotropic']):
                r0 = []  # as.list(1:object@p)
                for i in range(model['p']):
                    
                    r0.append(np.abs(testing_input[:,i][:,None] - model['input'][:,i]))
                    
                    #r0.append = as.matrix(abs(outer(testing_input[,i], object@input[,i], "-")))
                
            else:
                r0 = []
                if(p_x<model['num_obs']):
                    r0_here=0
                    for i in range(p_x):
                        
                        r0_here=r0_here+(np.abs(testing_input[:,i][:,None] - model['input'][:,i]))**2
                    
                    r0.append(np.sqrt(r0_here))
                else:
                    r0.append(euclidean_distance(testing_input,model['input']))
                    
        elif type(r0)== np.ndarray:
            r0_here=r0
            r0 = []
            r0.append(r0_here)
        elif type(r0) != list:
            sys.exit('r0 should be either a matrix or a list \n')
        # line 69
        
        if (len(r0)!=model['p']):
            sys.exit("the number of R0 matrices should be the same as the number of range parameters in the kernel \n")
          
        if ( (r0[0].shape[0]!=num_testing_input) or (r0[0].shape[1]!=model['num_obs'])):
            
            sys.exit("the dimension of R0 matrices should match the number of observations \n")
            
        kernel_type_num=np.repeat(0,model['p'])
        #rep(0,  object@p)
        
        for i_p in range(model['p']):
            if(model['kernel_type'][i_p]=="matern_5_2"):
                
                kernel_type_num[i_p]=3
            elif (model['kernel_type'][i_p]=="matern_3_2"):
                kernel_type_num[i_p]=2
            elif (model['kernel_type'][i_p]=="pow_exp"):
                kernel_type_num[i_p]=1
            elif (model['kernel_type'][i_p]=="periodic_gauss"):  ##this is periodic folding on Gaussian kernel
                kernel_type_num[i_p]=4
            elif (model['kernel_type'][i_p]=="periodic_exp"):   ##this is periodic folding on Exponential kernel
                kernel_type_num[i_p]=5
       
            
        if( (model['method']=='post_mode') or (model['method']=='mmle') ):
            
            #sp.stats.t.ppf(1 - alpha / 2, n - p - 1)           
           
            qt_025=sp.stats.t.ppf(0.025, model['num_obs'] - model['q'])  
            qt_975=sp.stats.t.ppf(0.975, model['num_obs'] - model['q'])  
            
            pred_list=pred_rgasp(model['beta_hat'],model['nugget'],model['input'],model['X'],model['zero_mean'],model['output'],
                                 testing_input,testing_trend,model['L'],model['LX'],model['theta_hat'],
                                 model['sigma2_hat'],qt_025,qt_975,r0,kernel_type_num,model['alpha'],model['method'],interval_data)
           
        
        elif model['method']=='mle':
            
            qn_025=sp.stats.norm.ppf(0.025)  
            qn_975=sp.stats.morn.ppf(0.975)  
            
            pred_list=pred_rgasp(model['beta_hat'],model['nugget'],model['input'],model['X'],model['zero_mean'],model['output'],
                                 testing_input,testing_trend,model['L'],model['LX'],model['theta_hat'],
                                 model['sigma2_hat'],qn_025,qn_975,r0,kernel_type_num,model['alpha'],model['method'],interval_data)
            
        output_list = {}
        
       
        output_list['mean']=pred_list[0]   #####can we all use @ or S? It will be more user friendly in that way 
        output_list['lower95']=pred_list[1]
        output_list['upper95']=pred_list[2]
        output_list['sd']=np.sqrt(pred_list[3]) 

        
        return output_list
        
        
  
        
        
                    
                
            
        
        
        
            
        
                    # line 336

                    
                        
                        
                        
    
                
                
            
            
                
                
                
                    
                
            
            
            
            
            
            
                  
           
            

            
            
            
            
            
        # LINE 505 END
                
                
              
              
        
                 
            
        
        
        
        
    

test = rgasp()
#design = np.random.normal(size = 50).reshape(50,1)
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


#response = np.random.normal(size = 50)
#task = test.create_task(design, response, nugget_est =True,method='mmle',prior_choice='ref_gamma',optimization= 'brent')
task = test.create_task(design, response)

model = test.train(task)

test_input = np.arange(0,10.01,1/100).reshape(-1,1)

result = test.predict_rgasp(model, 
                  test_input)

#print(higdon_1_data(4))








