<%
cfg['compiler_args'] = ['-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.7']
cfg['include_dirs'] = ['/usr/local/include/eigen']
setup_pybind11(cfg)
%>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/LU>
#include <Eigen/Dense>

namespace py = pybind11;

using namespace Eigen;
using namespace std;

typedef   Eigen::VectorXi        iVec;
typedef   Eigen::Map<iVec>      MapiVec;
typedef   Eigen::MatrixXd         Mat;
typedef   Eigen::Map<Mat>        MapMat;
typedef   Eigen::VectorXd         Vec;
typedef   Eigen::Map<Vec>        MapVec;
typedef   Eigen::ArrayXd          Ar1;
typedef   Eigen::Map<Ar1>        MapAr1;
typedef   Eigen::ArrayXXd         Ar2;
typedef   Eigen::Map<Ar2>        MapAr2;

Eigen::MatrixXd inv(Eigen::MatrixXd xs) {
    return xs.inverse();
}

double det(Eigen::MatrixXd xs) {
    return xs.determinant();
}

//const Eigen::Map<Eigen::MatrixXd> & d
Eigen::MatrixXd matern_5_2_funct(const Eigen::MatrixXd & d, double beta_i){
  //inline static Mat matern_5_2_funct (const Eigen::Map<Eigen::MatrixXd> & d, double beta_i){
  const double cnst = sqrt(5.0);
  Eigen::MatrixXd matOnes = Eigen::MatrixXd::Ones(d.rows(),d.cols());
  Eigen::MatrixXd result = cnst*beta_i*d;
  return ((matOnes + result +
	   result.array().pow(2.0).matrix()/3.0).cwiseProduct((-result).array().exp().matrix()));
  
}

 Eigen::MatrixXd matern_3_2_funct (const Eigen::MatrixXd & d, double beta_i){
  const double cnst = sqrt(3.0);
  Eigen::MatrixXd matOnes = Eigen::MatrixXd::Ones(d.rows(),d.cols());
  Eigen::MatrixXd result = cnst*beta_i*d;
  return ((matOnes + result ).cwiseProduct((-result).array().exp().matrix()));
  
}

Eigen::MatrixXd pow_exp_funct (const Eigen::MatrixXd & d, double beta_i,double alpha_i){
  
  return (-(beta_i*d).array().pow(alpha_i)).exp().matrix();

}

Eigen::MatrixXd periodic_gauss_funct(const Eigen::MatrixXd & d, double beta_i){
  
  int Rnrow = d.rows();
  int Rncol = d.cols();
  
  Eigen::MatrixXd R=1.0/(2.0*sqrt(M_PI*beta_i))*Eigen::MatrixXd::Ones(Rnrow,Rncol);
    double two_beta_i=2*beta_i;
    int n_ti=std::min(std::max(11.0, two_beta_i),101.0);
    for(int ti=1; ti <n_ti; ti++){
      R=R+1.0/sqrt(M_PI*beta_i)*exp(-pow(ti,2.0)/(4.0*beta_i))* (ti*d).array().cos().matrix();
  }
  R=R/R(0,0);
  return R;
  
  


}

Eigen::MatrixXd periodic_gauss_funct_fixed_normalized_const(const Eigen::MatrixXd & d, double beta_i,double perid_const_i){
  int Rnrow = d.rows();
  int Rncol = d.cols();
  
  Eigen::MatrixXd R=1.0/(2.0*sqrt(M_PI*beta_i))*Eigen::MatrixXd::Ones(Rnrow,Rncol);
  double two_beta_i=2*beta_i;
  int n_ti=std::min(std::max(11.0, two_beta_i),101.0);
  for(int ti=1; ti <n_ti; ti++){
    R=R+1.0/sqrt(M_PI*beta_i)*exp(-pow(ti,2.0)/(4.0*beta_i))* (ti*d).array().cos().matrix();
  }
  R=R/perid_const_i;
  return R;
  
}

Eigen::MatrixXd periodic_exp_funct(const Eigen::MatrixXd & d, double beta_i){
  
  int Rnrow = d.rows();
  int Rncol = d.cols();
  
  Eigen::MatrixXd R=1.0/(M_PI*beta_i)*Eigen::MatrixXd::Ones(Rnrow,Rncol);

  double five_beta_i=5*beta_i;
  int n_ti=std::min(std::max(21.0, five_beta_i),201.0);
  for(int ti=1; ti <n_ti; ti++){
    R=R+2.0*beta_i/((pow(beta_i,2.0)+pow(ti,2.0))*M_PI)*(ti*d).array().cos().matrix();
  }
  R=R/R(0,0);
  return R;
  

}

Eigen::MatrixXd periodic_exp_funct_fixed_normalized_const(const Eigen::MatrixXd & d, double beta_i,double perid_const_i){
  int Rnrow = d.rows();
  int Rncol = d.cols();
  
  Eigen::MatrixXd R=1.0/(M_PI*beta_i)*Eigen::MatrixXd::Ones(Rnrow,Rncol);
  
  double five_beta_i=5*beta_i;
  int n_ti=std::min(std::max(21.0, five_beta_i),201.0);
  for(int ti=1; ti <n_ti; ti++){
    R=R+2.0*beta_i/((pow(beta_i,2.0)+pow(ti,2.0))*M_PI)*(ti*d).array().cos().matrix();
  }
  R=R/perid_const_i;
  return R;
  
}

Eigen::MatrixXd  matern_5_2_deriv(const Eigen::MatrixXd & R0_i,  const Eigen::MatrixXd & R, double beta_i){
   
  const double sqrt_5 = sqrt(5.0);

  MatrixXd matOnes = Eigen::MatrixXd::Ones(R.rows(),R.cols());
  MatrixXd R0_i_2=R0_i.array().pow(2.0).matrix();

  MatrixXd part1= sqrt_5*R0_i+10.0/3*beta_i*R0_i_2;
  MatrixXd part2=matOnes+sqrt_5*beta_i*R0_i+5.0*pow(beta_i,2.0)*R0_i_2/3.0 ;
  return ((part1.cwiseQuotient(part2)  -sqrt_5*R0_i).cwiseProduct(R));
}

Eigen::MatrixXd  matern_3_2_deriv(const Eigen::MatrixXd & R0_i,  const Eigen::MatrixXd  & R, double beta_i){
   
  const double sqrt_3 = sqrt(3.0);

  return(-sqrt(3)*R0_i.cwiseProduct(R)+sqrt_3*R0_i.cwiseProduct((-sqrt_3*beta_i*R0_i).array().exp().matrix()));
    
}

Eigen::MatrixXd pow_exp_deriv(const Eigen::MatrixXd &  R0_i,  const Eigen::MatrixXd & R, const double beta_i, const double alpha_i){
 return  -(R.array()*(R0_i.array().pow(alpha_i))).matrix()*alpha_i*pow(beta_i,alpha_i-1);
}


Eigen::MatrixXd periodic_gauss_deriv(const Eigen::MatrixXd & R0_i, const Eigen::MatrixXd & R, double beta_i){
  
  int Rnrow = R0_i.rows();
  int Rncol = R0_i.cols();
  
  Eigen::MatrixXd R_here=1.0/(2.0*sqrt(M_PI*beta_i))*Eigen::MatrixXd::Ones(Rnrow,Rncol);
  
  Eigen::MatrixXd R_partial=-pow(beta_i,-1.5)/(4*sqrt(M_PI))*R.Ones(Rnrow,Rncol);
  double two_beta_i=2*beta_i;
  int n_ti=std::min(std::max(11.0, two_beta_i),101.0);
  for(int ti=1; ti <n_ti; ti++){
    R_here=R_here+1.0/sqrt(M_PI*beta_i)*exp(-pow(ti,2.0)/(4.0*beta_i))* (ti*R0_i).array().cos().matrix();
    R_partial=R_partial+ exp(-pow(ti,2.0)/(4.0*beta_i) )*pow(beta_i,-1.5)/(2*sqrt(M_PI))*(pow(ti,2.0)/(beta_i*2.0)-1 )*(ti*R0_i).array().cos().matrix();
  }
  
  double c_norm=R_here(0,0);
  double c_norm_partial=R_partial(0,0);
  
  return (R.array()*( R_partial.array()*c_norm/R_here.array()- (c_norm_partial*R.Ones(Rnrow,Rncol)).array() )/c_norm).matrix();
  
  
}

Eigen::MatrixXd periodic_exp_deriv(const Eigen::MatrixXd & R0_i, const Eigen::MatrixXd & R, double beta_i){
  
  int Rnrow = R0_i.rows();
  int Rncol = R0_i.cols();
  
  Eigen::MatrixXd R_here=1.0/(M_PI*beta_i)*Eigen::MatrixXd::Ones(Rnrow,Rncol);
  
  Eigen::MatrixXd R_partial=-1.0/(M_PI*pow(beta_i,2.0) )*R.Ones(Rnrow,Rncol);
  double five_beta_i=5*beta_i;
  int n_ti=std::min(std::max(21.0, five_beta_i),201.0);
  for(int ti=1; ti <n_ti; ti++){
    R_here=R_here+2.0*beta_i/((pow(beta_i,2.0)+pow(ti,2.0))*M_PI)*(ti*R0_i).array().cos().matrix();
    R_partial=R_partial+ 2.0*(pow(ti,2.0)-pow(beta_i,2.0) )/(M_PI*(pow(beta_i,2.0)+pow(ti,2.0) ))*(ti*R0_i).array().cos().matrix();
  }
  
  double c_norm=R_here(0,0);
  double c_norm_partial=R_partial(0,0);
  
  return (R.array()*( R_partial.array()*c_norm/R_here.array()- (c_norm_partial*R.Ones(Rnrow,Rncol)).array() )/c_norm).matrix();
  
}

Eigen::MatrixXd separable_kernel (const py::list& R0, const Eigen::VectorXd  & beta, std::string kernel_type, const Eigen::VectorXd  & alpha ){
  //Eigen::MatrixXd R0element = R0[0];
  Eigen::MatrixXd R0element = R0[0].cast<Eigen::MatrixXd>();
  int Rnrow = R0element.rows();
  int Rncol = R0element.cols();
  
  Eigen::MatrixXd R = R.Ones(Rnrow,Rncol);
  if(kernel_type=="matern_5_2"){
    for (int i_ker = 0; i_ker < beta.size(); i_ker++){
      //R = (matern_5_2_funct(R0[i_ker],beta[i_ker])).cwiseProduct(R);
      R = (matern_5_2_funct(R0[i_ker].cast<Eigen::MatrixXd>(),beta[i_ker])).cwiseProduct(R);
    }
  }else if(kernel_type=="matern_3_2"){
    for (int i_ker = 0; i_ker < beta.size(); i_ker++){
      R = (matern_3_2_funct(R0[i_ker].cast<Eigen::MatrixXd>(),beta[i_ker])).cwiseProduct(R);
    }
  }
  else if(kernel_type=="pow_exp"){
    for (int i_ker = 0; i_ker < beta.size(); i_ker++){
      R = (pow_exp_funct(R0[i_ker].cast<Eigen::MatrixXd>(),beta[i_ker],alpha[i_ker])).cwiseProduct(R);
    }
  }
  else if(kernel_type=="periodic_gauss"){
    for (int i_ker = 0; i_ker < beta.size(); i_ker++){
      R = (periodic_gauss_funct(R0[i_ker].cast<Eigen::MatrixXd>(),beta[i_ker])).cwiseProduct(R);
    }
  }
  else if(kernel_type=="periodic_exp"){
    for (int i_ker = 0; i_ker < beta.size(); i_ker++){
      R = (periodic_exp_funct(R0[i_ker].cast<Eigen::MatrixXd>(),beta[i_ker])).cwiseProduct(R);
    }
  }
  
  
  return R;
}


Eigen::MatrixXd separable_multi_kernel (const py::list& R0, const Eigen::VectorXd  & beta,const Eigen::VectorXi  & kernel_type,const Eigen::VectorXd  & alpha ){
  Eigen::MatrixXd R0element = R0[0].cast<Eigen::MatrixXd>();
  int Rnrow = R0element.rows();
  int Rncol = R0element.cols();

  Eigen::MatrixXd R = R.Ones(Rnrow,Rncol);
  //String kernel_type_i_ker;
  for (int i_ker = 0; i_ker < beta.size(); i_ker++){
   // kernel_type_i_ker=kernel_type[i_ker];
    if(kernel_type[i_ker]==3){
      R = (matern_5_2_funct(R0[i_ker].cast<Eigen::MatrixXd>(),beta[i_ker])).cwiseProduct(R);
    }else if(kernel_type[i_ker]==2){
      R = (matern_3_2_funct(R0[i_ker].cast<Eigen::MatrixXd>(),beta[i_ker])).cwiseProduct(R);
    }else if(kernel_type[i_ker]==1){
      R = (pow_exp_funct(R0[i_ker].cast<Eigen::MatrixXd>(),beta[i_ker],alpha[i_ker])).cwiseProduct(R);
    }else if(kernel_type[i_ker]==4){
      R = (periodic_gauss_funct(R0[i_ker].cast<Eigen::MatrixXd>(),beta[i_ker])).cwiseProduct(R);
    }else if(kernel_type[i_ker]==5){
      R = (periodic_exp_funct(R0[i_ker].cast<Eigen::MatrixXd>(),beta[i_ker])).cwiseProduct(R);
    }
      
  }
  return R;
}

Eigen::MatrixXd separable_multi_kernel_pred_periodic (const py::list& R0, const Eigen::VectorXd &  beta,const Eigen::VectorXi  & kernel_type, const Eigen::VectorXd  & alpha, const Eigen::VectorXd  & perid_const){
  Eigen::MatrixXd R0element = R0[0].cast<Eigen::MatrixXd>();
  int Rnrow = R0element.rows();
  int Rncol = R0element.cols();
  
  Eigen::MatrixXd R = R.Ones(Rnrow,Rncol);
  //String kernel_type_i_ker;
  for (int i_ker = 0; i_ker < beta.size(); i_ker++){
    // kernel_type_i_ker=kernel_type[i_ker];
    if(kernel_type[i_ker]==3){
      R = (matern_5_2_funct(R0[i_ker].cast<Eigen::MatrixXd>(),beta[i_ker])).cwiseProduct(R);
    }else if(kernel_type[i_ker]==2){
      R = (matern_3_2_funct(R0[i_ker].cast<Eigen::MatrixXd>(),beta[i_ker])).cwiseProduct(R);
    }else if(kernel_type[i_ker]==1){
      R = (pow_exp_funct(R0[i_ker].cast<Eigen::MatrixXd>(),beta[i_ker],alpha[i_ker])).cwiseProduct(R);
    }else if(kernel_type[i_ker]==4){
      R = (periodic_gauss_funct_fixed_normalized_const(R0[i_ker].cast<Eigen::MatrixXd>(),beta[i_ker],perid_const[i_ker])).cwiseProduct(R);
    }else if(kernel_type[i_ker]==5){
      R = (periodic_exp_funct_fixed_normalized_const(R0[i_ker].cast<Eigen::MatrixXd>(),beta[i_ker],perid_const[i_ker])).cwiseProduct(R);
    }
  }
  return R;
}

Eigen::MatrixXd euclidean_distance(const Eigen::MatrixXd & input1,const Eigen::MatrixXd & input2){
  //input are n by p, where p is larger than n
  
  int num_obs1 = input1.rows();
  int num_obs2 = input2.rows();
  
  Eigen::MatrixXd R0=R0.Ones(num_obs1,num_obs2);
  
  for (int i = 0; i < num_obs1; i++){
    
    for (int j = 0; j < num_obs2; j++){
      R0(i,j)=sqrt((input1.row(i)-input2.row(j)).array().pow(2.0).sum());
    }
  }
  return R0;
}

double log_marginal_lik(const Eigen::VectorXd  & param,double nugget, const bool nugget_est, const py::list& R0, const Eigen::MatrixXd & X,const  std::string zero_mean,const Eigen::MatrixXd & output,const Eigen::VectorXi & kernel_type,const Eigen::VectorXd & alpha ){
  Eigen::VectorXd beta;
  double nu=nugget;
  int param_size=param.size();
  if(!nugget_est){
    beta= param.array().exp().matrix();
    // nu=0;
  }else{
    beta=param.head(param_size-1).array().exp().matrix(); 
    nu=exp(param[param_size-1]); //nugget
  }

  int num_obs=output.rows();
  MatrixXd R= separable_multi_kernel(R0,beta, kernel_type,alpha);
  R=R+nu*MatrixXd::Identity(num_obs,num_obs);  //not sure 
	       
  //Eigen::MatrixXd L( R.llt().matrixL() );
  //Eigen::MatrixXd L_T=L.adjoint();	    
     
  Eigen::LLT<MatrixXd> lltOfR(R);             // compute the cholesky decomposition of R called lltofR
  MatrixXd L = lltOfR.matrixL();   //retrieve factor L  in the decomposition

  if(zero_mean=="Yes"){

   MatrixXd yt_R_inv= (L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(output))).transpose(); 
  MatrixXd S_2= (yt_R_inv*output);
  double log_S_2=log(S_2(0,0));
    
   return (-(L.diagonal().array().log().matrix().sum())-(num_obs)/2.0*log_S_2);
   
  }else{
    
  int q=X.cols();

  MatrixXd R_inv_X=L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(X)); //one forward and one backward to compute R.inv%*%X
  MatrixXd Xt_R_inv_X=X.transpose()*R_inv_X; //Xt%*%R.inv%*%X

  Eigen::LLT<MatrixXd> lltOfXRinvX(Xt_R_inv_X); // cholesky decomposition of Xt_R_inv_X called lltOfXRinvX
  MatrixXd LX = lltOfXRinvX.matrixL();  //  retrieve factor LX  in the decomposition 
  
  //Eigen::MatrixXd LX( Xt_R_inv_X.llt().matrixL() );
  //Eigen::MatrixXd L_T=L.adjoint();	 
  
  MatrixXd R_inv_X_Xt_R_inv_X_inv_Xt_R_inv= R_inv_X*(LX.transpose().triangularView<Upper>().solve(LX.triangularView<Lower>().solve(R_inv_X.transpose())));          //compute  R_inv_X_Xt_R_inv_X_inv_Xt_R_inv through one forward and one backward solve
  MatrixXd yt_R_inv= (L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(output))).transpose(); 
  MatrixXd S_2= (yt_R_inv*output-output.transpose()*R_inv_X_Xt_R_inv_X_inv_Xt_R_inv*output);
  double log_S_2=log(S_2(0,0));
  return (-(L.diagonal().array().log().matrix().sum())-(LX.diagonal().array().log().matrix().sum())-(num_obs-q)/2.0*log_S_2);
  }
}

double log_profile_lik(const Eigen::VectorXd & param,double nugget, const bool nugget_est, const py::list& R0, const Eigen::MatrixXd  & X,const std::string zero_mean,const Eigen::MatrixXd & output,const Eigen::VectorXi & kernel_type,const Eigen::VectorXd & alpha ){
  Eigen::VectorXd beta;
  double nu=nugget;
  int param_size=param.size();
  if(!nugget_est){
    beta= param.array().exp().matrix();
    // nu=0;
  }else{
    beta=param.head(param_size-1).array().exp().matrix(); 
    nu=exp(param[param_size-1]); //nugget
  }
  
  int num_obs=output.rows();
  MatrixXd R= separable_multi_kernel(R0,beta, kernel_type,alpha);
  R=R+nu*MatrixXd::Identity(num_obs,num_obs);  //not sure 
  
  LLT<MatrixXd> lltOfR(R);             // compute the cholesky decomposition of R called lltofR
  MatrixXd L = lltOfR.matrixL();   //retrieve factor L  in the decomposition
  
  if(zero_mean=="Yes"){
    
    MatrixXd yt_R_inv= (L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(output))).transpose(); 
    MatrixXd S_2= (yt_R_inv*output);
    double log_S_2=log(S_2(0,0));
    
    return (-(L.diagonal().array().log().matrix().sum())-(num_obs)/2.0*log_S_2);
    
  }else{
    
    //int q=X.cols();
    
    MatrixXd R_inv_X=L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(X)); //one forward and one backward to compute R.inv%*%X
    MatrixXd Xt_R_inv_X=X.transpose()*R_inv_X; //Xt%*%R.inv%*%X
    
    LLT<MatrixXd> lltOfXRinvX(Xt_R_inv_X); // cholesky decomposition of Xt_R_inv_X called lltOfXRinvX
    MatrixXd LX = lltOfXRinvX.matrixL();  //  retrieve factor LX  in the decomposition 
    MatrixXd R_inv_X_Xt_R_inv_X_inv_Xt_R_inv= R_inv_X*(LX.transpose().triangularView<Upper>().solve(LX.triangularView<Lower>().solve(R_inv_X.transpose())));          //compute  R_inv_X_Xt_R_inv_X_inv_Xt_R_inv through one forward and one backward solve
    MatrixXd yt_R_inv= (L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(output))).transpose(); 
    MatrixXd S_2= (yt_R_inv*output-output.transpose()*R_inv_X_Xt_R_inv_X_inv_Xt_R_inv*output);
    double log_S_2=log(S_2(0,0));
    
    //change to profile likelihood seems only needs one step
    //return (-(L.diagonal().array().log().matrix().sum())-(LX.diagonal().array().log().matrix().sum())-(num_obs-q)/2.0*log_S_2);
    return (-(L.diagonal().array().log().matrix().sum())-(num_obs)/2.0*log_S_2);
    
  }
}

double log_approx_ref_prior(const Eigen::VectorXd & param,double nugget, bool nugget_est, const Eigen::VectorXd & CL,const double a,const double b ){

  Eigen::VectorXd beta;
  double nu=nugget;
  int param_size=param.size();
  if(!nugget_est){
    beta= param.array().exp().matrix();
  }else{
    beta=param.head(param_size-1).array().exp().matrix(); 
    nu=exp(param[param_size-1]); //nugget
  }
  double t=CL.cwiseProduct(beta).sum()+nu;
  double part_I=-b*t;
  double part_II= a*log(t);
  return part_I+part_II;
}

Eigen::VectorXd log_marginal_lik_deriv(const Eigen::VectorXd & param,double nugget,  bool nugget_est, const py::list& R0, const Eigen::MatrixXd & X,const std::string zero_mean,const Eigen::MatrixXd & output, Eigen::VectorXi & kernel_type,const Eigen::VectorXd & alpha){
    
  Eigen::VectorXd beta;
  double nu=nugget;
  int param_size=param.size();
  if(nugget_est==false){//not sure about the logical stuff
    beta= param.array().exp().matrix();
  }else{
    beta=param.head(param_size-1).array().exp().matrix(); 
    nu=exp(param[param_size-1]); //nugget
  }
  int p=beta.size();
  int num_obs=output.rows();
  MatrixXd R= separable_multi_kernel(R0,beta,kernel_type,alpha);
  MatrixXd R_ori=  R;  // this is the one without the nugget
    
  R=R+nu*MatrixXd::Identity(num_obs,num_obs);  //not sure 
    
  LLT<MatrixXd> lltOfR(R);
  MatrixXd L = lltOfR.matrixL();
  VectorXd ans=VectorXd::Ones(param_size);
  
  //String kernel_type_ti;
  
  if(zero_mean=="Yes"){
    MatrixXd yt_R_inv= (L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(output))).transpose(); 
    MatrixXd S_2= (yt_R_inv*output);
    //double log_S_2=log(S_2(0,0));

    
    MatrixXd dev_R_i;
    MatrixXd Vb_ti;
    //allow different choices of kernels
    for(int ti=0;ti<p;ti++){
      //kernel_type_ti=kernel_type[ti];
      if(kernel_type[ti]==3){
        dev_R_i=matern_5_2_deriv( R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti]);  //now here I have R_ori instead of R
      }else if(kernel_type[ti]==2){
        dev_R_i=matern_3_2_deriv( R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti]);  //now here I have R_ori instead of R
      }else if(kernel_type[ti]==1){
        dev_R_i=pow_exp_deriv( R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti],alpha[ti]);   //now here I have R_ori instead of R
      }else if(kernel_type[ti]==4){
        dev_R_i = (periodic_gauss_deriv(R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti])).cwiseProduct(R);
      }else if(kernel_type[ti]==5){
        dev_R_i = (periodic_exp_deriv(R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti])).cwiseProduct(R);
      }
      Vb_ti=L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(dev_R_i));
      ans[ti]=-0.5*Vb_ti.diagonal().sum()+(num_obs/2.0*output.transpose()*Vb_ti*yt_R_inv.transpose()/ S_2(0,0))(0,0) ;  
    }
    //the last one if the nugget exists
    if(nugget_est){
      dev_R_i=MatrixXd::Identity(num_obs,num_obs);
      Vb_ti=L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(dev_R_i));
      ans[p]=-0.5*Vb_ti.diagonal().sum()+(num_obs/2.0*output.transpose()*Vb_ti*yt_R_inv.transpose()/ S_2(0,0))(0,0); 
    }
    
  }else{
    int q=X.cols();
    MatrixXd R_inv_X=L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(X));
    MatrixXd Xt_R_inv_X=X.transpose()*R_inv_X;

    LLT<MatrixXd> lltOfXRinvX(Xt_R_inv_X);
    MatrixXd LX = lltOfXRinvX.matrixL();
    MatrixXd R_inv_X_Xt_R_inv_X_inv_Xt_R_inv= R_inv_X*(LX.transpose().triangularView<Upper>().solve(LX.triangularView<Lower>().solve(R_inv_X.transpose())));
    MatrixXd yt_R_inv= (L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(output))).transpose();
    MatrixXd S_2= (yt_R_inv*output-output.transpose()*R_inv_X_Xt_R_inv_X_inv_Xt_R_inv*output);

    MatrixXd Q_output= yt_R_inv.transpose()-R_inv_X_Xt_R_inv_X_inv_Xt_R_inv*output;
    MatrixXd dev_R_i;
    MatrixXd Wb_ti;
    //allow different choices of kernels
    
    for(int ti=0;ti<p;ti++){
      //kernel_type_ti=kernel_type[ti];
      if(kernel_type[ti]==3){
        dev_R_i=matern_5_2_deriv( R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti]);  //now here I have R_ori instead of R
      }else if(kernel_type[ti]==2){
        dev_R_i=matern_3_2_deriv( R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti]);  //now here I have R_ori instead of R
      }else if(kernel_type[ti]==1){
        dev_R_i=pow_exp_deriv( R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti],alpha[ti]);   //now here I have R_ori instead of R
      }else if(kernel_type[ti]==4){
        dev_R_i = (periodic_gauss_deriv(R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti])).cwiseProduct(R);
      }else if(kernel_type[ti]==5){
        dev_R_i = (periodic_exp_deriv(R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti])).cwiseProduct(R);
      }
      Wb_ti=(L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(dev_R_i))).transpose()-dev_R_i*R_inv_X_Xt_R_inv_X_inv_Xt_R_inv;
      ans[ti]=-0.5*Wb_ti.diagonal().sum()+(num_obs-q)/2.0*(output.transpose()*Wb_ti.transpose()*Q_output/S_2(0,0))(0,0); 
    }
    
    if(nugget_est){
      dev_R_i=MatrixXd::Identity(num_obs,num_obs);
      Wb_ti=(L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(dev_R_i))).transpose()-dev_R_i*R_inv_X_Xt_R_inv_X_inv_Xt_R_inv;
      ans[p]=-0.5*Wb_ti.diagonal().sum()+(num_obs-q)/2.0*(output.transpose()*Wb_ti.transpose()*Q_output/S_2(0,0))(0,0); 
    }
    

  }
      return ans;

}

Eigen::VectorXd log_profile_lik_deriv(const Eigen::VectorXd & param,double nugget,  bool nugget_est, const  const py::list& R0, const Eigen::MatrixXd & X,const std::string zero_mean,const Eigen::MatrixXd & output,   const Eigen::VectorXi & kernel_type,const Eigen::VectorXd & alpha){
  
  Eigen::VectorXd beta;
  double nu=nugget;
  int param_size=param.size();
  if(nugget_est==false){//not sure about the logical stuff
    beta= param.array().exp().matrix();
  }else{
    beta=param.head(param_size-1).array().exp().matrix(); 
    nu=exp(param[param_size-1]); //nugget
  }
  int p=beta.size();
  int num_obs=output.rows();
  MatrixXd R= separable_multi_kernel(R0,beta,kernel_type,alpha);
  MatrixXd R_ori=  R;  // this is the one without the nugget
  
  R=R+nu*MatrixXd::Identity(num_obs,num_obs);  //not sure 
  
  LLT<MatrixXd> lltOfR(R);
  MatrixXd L = lltOfR.matrixL();
  VectorXd ans=VectorXd::Ones(param_size);
  
  //String kernel_type_ti;
  MatrixXd Vb_ti;
  
  if(zero_mean=="Yes"){
    MatrixXd yt_R_inv= (L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(output))).transpose(); 
    MatrixXd S_2= (yt_R_inv*output);
    //double log_S_2=log(S_2(0,0));
    
    
    MatrixXd dev_R_i;
    //allow different choices of kernels
    for(int ti=0;ti<p;ti++){
      //kernel_type_ti=kernel_type[ti];
      if(kernel_type[ti]==3){
        dev_R_i=matern_5_2_deriv( R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti]);  //now here I have R_ori instead of R
      }else if(kernel_type[ti]==2){
        dev_R_i=matern_3_2_deriv( R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti]);  //now here I have R_ori instead of R
      }else if(kernel_type[ti]==1){
        dev_R_i=pow_exp_deriv( R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti],alpha[ti]);   //now here I have R_ori instead of R
      }else if(kernel_type[ti]==4){
        dev_R_i = (periodic_gauss_deriv(R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti])).cwiseProduct(R);
      }else if(kernel_type[ti]==5){
        dev_R_i = (periodic_exp_deriv(R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti])).cwiseProduct(R);
      }
      Vb_ti=L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(dev_R_i));
      ans[ti]=-0.5*Vb_ti.diagonal().sum()+(num_obs/2.0*output.transpose()*Vb_ti*yt_R_inv.transpose()/ S_2(0,0))(0,0) ;  
    }
    //the last one if the nugget exists
    if(nugget_est){
      dev_R_i=MatrixXd::Identity(num_obs,num_obs);
      Vb_ti=L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(dev_R_i));
      ans[p]=-0.5*Vb_ti.diagonal().sum()+(num_obs/2.0*output.transpose()*Vb_ti*yt_R_inv.transpose()/ S_2(0,0))(0,0); 
    }
    
  }else{
   // int q=X.cols();
    MatrixXd R_inv_X=L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(X));
    MatrixXd Xt_R_inv_X=X.transpose()*R_inv_X;
    
    LLT<MatrixXd> lltOfXRinvX(Xt_R_inv_X);
    MatrixXd LX = lltOfXRinvX.matrixL();
    MatrixXd R_inv_X_Xt_R_inv_X_inv_Xt_R_inv= R_inv_X*(LX.transpose().triangularView<Upper>().solve(LX.triangularView<Lower>().solve(R_inv_X.transpose())));
    MatrixXd yt_R_inv= (L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(output))).transpose();
    MatrixXd S_2= (yt_R_inv*output-output.transpose()*R_inv_X_Xt_R_inv_X_inv_Xt_R_inv*output);
    
    MatrixXd Q_output= yt_R_inv.transpose()-R_inv_X_Xt_R_inv_X_inv_Xt_R_inv*output;
    MatrixXd dev_R_i;
    MatrixXd Wb_ti;
    //allow different choices of kernels
    
    for(int ti=0;ti<p;ti++){
      //kernel_type_ti=kernel_type[ti];
      if(kernel_type[ti]==3){
        dev_R_i=matern_5_2_deriv( R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti]);  //now here I have R_ori instead of R
      }else if(kernel_type[ti]==2){
        dev_R_i=matern_3_2_deriv( R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti]);  //now here I have R_ori instead of R
      }else if(kernel_type[ti]==1){
        dev_R_i=pow_exp_deriv( R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti],alpha[ti]);   //now here I have R_ori instead of R
      }else if(kernel_type[ti]==4){
        dev_R_i = (periodic_gauss_deriv(R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti])).cwiseProduct(R);
      }else if(kernel_type[ti]==5){
        dev_R_i = (periodic_exp_deriv(R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti])).cwiseProduct(R);
      }
      Vb_ti=(L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(dev_R_i)));
      Wb_ti=Vb_ti.transpose()-dev_R_i*R_inv_X_Xt_R_inv_X_inv_Xt_R_inv;
      ans[ti]=-0.5*Vb_ti.diagonal().sum()+(num_obs)/2.0*(output.transpose()*Wb_ti.transpose()*Q_output/S_2(0,0))(0,0); 
    }
    
    if(nugget_est){
      dev_R_i=MatrixXd::Identity(num_obs,num_obs);
      Vb_ti=(L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(dev_R_i)));
      Wb_ti=Vb_ti.transpose()-dev_R_i*R_inv_X_Xt_R_inv_X_inv_Xt_R_inv;
      ans[p]=-0.5*Vb_ti.diagonal().sum()+(num_obs)/2.0*(output.transpose()*Wb_ti.transpose()*Q_output/S_2(0,0))(0,0); 
      
      //Vb_ti=L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(dev_R_i));
      //ans[p]=-0.5*Vb_ti.diagonal().sum()+(num_obs/2.0*output.transpose()*Vb_ti*yt_R_inv.transpose()/ S_2(0,0))(0,0); 
      
    }
    
    
  }
  return ans;
  
}

Eigen::VectorXd log_approx_ref_prior_deriv(const Eigen::VectorXd & param,double nugget, bool nugget_est, const Eigen::VectorXd & CL,const double a,const double b ){

  Eigen::VectorXd beta;
  Eigen::VectorXd return_vec;
  double nu=nugget;
  int param_size=param.size();
  if(!nugget_est){//not sure about the logical stuff. Previously (nugget_est==false)
    beta= param.array().exp().matrix();
  }else{
    beta=param.head(param_size-1).array().exp().matrix(); 
    nu=exp(param[param_size-1]); //nugget
  }

  //  double a=1/2.0;//let people specify
  // double b=(a+beta.size())/2.0;
  double t=CL.cwiseProduct(beta).sum()+nu;

  if(!nugget_est){
    return_vec=(a*CL/t- b*CL);
  }else{
    Eigen::VectorXd CL_1(param_size);
    CL_1.head(param_size-1)=CL;
    CL_1[param_size-1]=1;
    return_vec=(a*CL_1/t- b*CL_1);
  }
  return return_vec;

}

double log_ref_marginal_post(const Eigen::VectorXd & param,double nugget, bool nugget_est, const py::list& R0, const Eigen::MatrixXd & X,const std::string zero_mean,const Eigen::MatrixXd & output, const Eigen::VectorXi & kernel_type,const Eigen::VectorXd & alpha){
    
  Eigen::VectorXd beta;
  double nu=nugget;
  int param_size=param.size();
  if(nugget_est==false){//not sure about the logical stuff
    beta= param.array().exp().matrix();
  }else{
    beta=param.head(param_size-1).array().exp().matrix(); 
    nu=exp(param[param_size-1]); //nugget
  }
  int p=beta.size();
  int num_obs=output.rows();
  MatrixXd R= separable_multi_kernel(R0,beta,kernel_type,alpha);
  MatrixXd R_ori=  R;  // this is the one without the nugget
    
  R=R+nu*MatrixXd::Identity(num_obs,num_obs);  //not sure 
    
  LLT<MatrixXd> lltOfR(R);
  MatrixXd L = lltOfR.matrixL();

 // String kernel_type_ti;
  
  if(zero_mean=="Yes"){

    MatrixXd yt_R_inv= (L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(output))).transpose(); 
    MatrixXd S_2= (yt_R_inv*output);
    double log_S_2=log(S_2(0,0));

    
    VectorXd ans=VectorXd::Ones(param_size);
    MatrixXd dev_R_i;
    py::list Vb(param_size);
    //allow different choices of kernels
    for(int ti=0;ti<p;ti++){
    //  kernel_type_ti=kernel_type[ti];
      if(kernel_type[ti]==3){
        dev_R_i=matern_5_2_deriv( R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti]);  //now here I have R_ori instead of R
      }else if(kernel_type[ti]==2){
        dev_R_i=matern_3_2_deriv( R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti]);  //now here I have R_ori instead of R
      }else if(kernel_type[ti]==1){
        dev_R_i=pow_exp_deriv( R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti],alpha[ti]);   //now here I have R_ori instead of R
      }else if(kernel_type[ti]==4){
        dev_R_i = (periodic_gauss_deriv(R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti])).cwiseProduct(R);
      }else if(kernel_type[ti]==5){
        dev_R_i = (periodic_exp_deriv(R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti])).cwiseProduct(R);
      }
      Vb[ti].cast<Eigen::MatrixXd>()=L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(dev_R_i));
    }

    //the last one if the nugget exists
    if(nugget_est){
      dev_R_i=MatrixXd::Identity(num_obs,num_obs);
      Vb[param_size-1].cast<Eigen::MatrixXd>()=L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(dev_R_i));
    }
  // int q=X.cols();
  MatrixXd IR(param_size+1,param_size+1);
  IR(0,0)=num_obs;
  
  for(int i=0;i<param_size;i++){
    MatrixXd Vb_i=Vb[i].cast<Eigen::MatrixXd>();
    IR(0,i+1)=IR(i+1,0)= Vb_i.trace();
    for(int j=0;j<param_size;j++){
      MatrixXd Vb_j=Vb[j].cast<Eigen::MatrixXd>();
      IR(i+1,j+1)=IR(j+1,i+1)=(Vb_i*Vb_j).trace();

    }
  }

  LLT<MatrixXd> lltOfIR(IR);
  MatrixXd LIR = lltOfIR.matrixL();
    
  return (-(L.diagonal().array().log().matrix().sum())-(num_obs)/2.0*log_S_2+ LIR.diagonal().array().log().matrix().sum());
  }else{
  int q=X.cols();
  MatrixXd R_inv_X=L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(X));
  MatrixXd Xt_R_inv_X=X.transpose()*R_inv_X;

  LLT<MatrixXd> lltOfXRinvX(Xt_R_inv_X);
  MatrixXd LX = lltOfXRinvX.matrixL();
  MatrixXd R_inv_X_Xt_R_inv_X_inv_Xt_R_inv= R_inv_X*(LX.transpose().triangularView<Upper>().solve(LX.triangularView<Lower>().solve(R_inv_X.transpose())));
  MatrixXd yt_R_inv= (L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(output))).transpose();
  MatrixXd S_2= (yt_R_inv*output-output.transpose()*R_inv_X_Xt_R_inv_X_inv_Xt_R_inv*output);

  MatrixXd Q_output= yt_R_inv.transpose()-R_inv_X_Xt_R_inv_X_inv_Xt_R_inv*output;
  MatrixXd dev_R_i;
  py::list Wb(param_size);
  
  
  for(int ti=0;ti<p;ti++){
  //  kernel_type_ti=kernel_type[ti];
    if(kernel_type[ti]==3){
      dev_R_i=matern_5_2_deriv( R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti]);  //now here I have R_ori instead of R
    }else if(kernel_type[ti]==2){
      dev_R_i=matern_3_2_deriv( R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti]);  //now here I have R_ori instead of R
    }else if(kernel_type[ti]==1){
      dev_R_i=pow_exp_deriv( R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti],alpha[ti]);   //now here I have R_ori instead of R
    }else if(kernel_type[ti]==4){
      dev_R_i = (periodic_gauss_deriv(R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti])).cwiseProduct(R);
    }else if(kernel_type[ti]==5){
      dev_R_i = (periodic_exp_deriv(R0[ti].cast<Eigen::MatrixXd>(),R_ori,beta[ti])).cwiseProduct(R);
    }
    Wb[ti].cast<Eigen::MatrixXd>()=(L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(dev_R_i))).transpose()-dev_R_i*R_inv_X_Xt_R_inv_X_inv_Xt_R_inv;
  }
  
  
  //the last one if the nugget exists
  if(nugget_est){
    dev_R_i=MatrixXd::Identity(num_obs,num_obs);
    Wb[param_size-1].cast<Eigen::MatrixXd>()=(L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(dev_R_i))).transpose()-dev_R_i*R_inv_X_Xt_R_inv_X_inv_Xt_R_inv;
  }
  MatrixXd IR(param_size+1,param_size+1);
  IR(0,0)=num_obs-q;
  for(int i=0;i<param_size;i++){
    MatrixXd Wb_i=Wb[i].cast<Eigen::MatrixXd>();
    IR(0,i+1)=IR(i+1,0)= Wb_i.trace();
    for(int j=0;j<param_size;j++){
      MatrixXd Wb_j=Wb[j].cast<Eigen::MatrixXd>();
      IR(i+1,j+1)=IR(j+1,i+1)=(Wb_i*Wb_j).trace();

    }
  }

  LLT<MatrixXd> lltOfIR(IR);
  MatrixXd LIR = lltOfIR.matrixL();

  double log_S_2=log(S_2(0,0));
    
  return (-(L.diagonal().array().log().matrix().sum())-(LX.diagonal().array().log().matrix().sum())-(num_obs-q)/2.0*log_S_2+ LIR.diagonal().array().log().matrix().sum());
  }
  //  return (-(L.diagonal().array().log().matrix().sum())-(LX.diagonal().array().log().matrix().sum())-(num_obs-q)/2.0*log_S_2+1/2.0*log(IR.determinant()) );
}



PYBIND11_PLUGIN(my_own) {
    pybind11::module m("my_own", "auto-compiled c++ extension");
    m.def("inv", &inv);
    m.def("det", &det);
    m.def("matern_5_2_funct", &matern_5_2_funct);
    m.def("matern_3_2_funct", &matern_3_2_funct);
    m.def("pow_exp_funct", &pow_exp_funct); 
    m.def("periodic_gauss_funct", &periodic_gauss_funct);
    m.def("periodic_gauss_funct_fixed_normalized_const", &periodic_gauss_funct_fixed_normalized_const);
    m.def("periodic_exp_funct", &periodic_exp_funct);
    m.def("periodic_exp_funct_fixed_normalized_const", &periodic_exp_funct_fixed_normalized_const);
    m.def("matern_5_2_deriv", &matern_5_2_deriv);
    m.def("matern_3_2_deriv", &matern_3_2_deriv);
    m.def("pow_exp_deriv", &pow_exp_deriv);
    m.def("periodic_gauss_deriv", &periodic_gauss_deriv);
    m.def("periodic_exp_deriv", &periodic_exp_deriv);  //line 214 functions.cp 
    m.def("separable_kernel", &separable_kernel);
    m.def("separable_multi_kernel", &separable_multi_kernel);
    m.def("separable_multi_kernel_pred_periodic", &separable_multi_kernel_pred_periodic);
    m.def("euclidean_distance", &euclidean_distance);
    m.def("log_marginal_lik", &log_marginal_lik);
    m.def("log_profile_lik", &log_profile_lik);
    m.def("log_approx_ref_prior", &log_approx_ref_prior);
    m.def("log_marginal_lik_deriv", &log_marginal_lik_deriv);
    m.def("log_profile_lik_deriv", &log_profile_lik_deriv);
    
    m.def("log_approx_ref_prior_deriv", &log_approx_ref_prior_deriv);    
    m.def("log_ref_marginal_post", &log_ref_marginal_post);  //line 817
    
    
//     m.def("matern_3_2_funct", &matern_3_2_funct);
//     m.def("matern_3_2_funct", &matern_3_2_funct);
//     m.def("matern_3_2_funct", &matern_3_2_funct);
    
//     m.def("matern_3_2_funct", &matern_3_2_funct);
//     m.def("matern_3_2_funct", &matern_3_2_funct);
    
    return m.ptr();
}