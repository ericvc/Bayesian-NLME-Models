functions{
  
  real logistic3(real x, vector theta){
    /* 
    -3 parameter logistic function-
    theta[1] is the upper y-asymptote
    theta[2] is the x-coordinate of the first inflection point
    theta[3] is the rate parameter (stepness of curve)
    */
    real y;
    y = theta[1] / (1+exp((theta[2]-x)/theta[3]));
    return y;
  }
  
  real logistic5(real x, vector theta){
    /* 
    -5 parameter logistic function-
    theta[1] is the upper y-asym-ptote
    theta[2] is the x-coordinate of the first inflection point
    theta[3] is the first growth rate parameter
    theta[4] is the x-coordinate of the second inflection point
    theta[5] is the second growth rate parameter
    */
    real y;
    y = theta[1] / (1+exp((theta[2]-x)/theta[3])) - 
        theta[1] / (1+exp((theta[4]-x)/theta[5]));
    return y;
  }
  
  real logistic6(real x, vector theta){
    /*
    -6 parameter logistic function-
    theta[1] is the upper y-asymptote
    theta[2] is the x-coordinate of the first inflection point
    theta[3] is the first growth rate parameter
    theta[4] is the x-coordinate of the second inflection point
    theta[5] is the second growth rate parameter
    theta[6] is related to the value of the second lower asymptote
    */
    real y;
    y = theta[1] / (1+exp((theta[2]-x)/theta[3])) - 
        (theta[1]*theta[6]) / (1+exp((theta[4]-x)/theta[5]));
    return y;
  }

  real logisticN(real x, vector theta){
    /* 
    Wrapper function for all NLME models. The size of the parameter vector is checked, 
    which determines the function to call.
    */
    int K = rows(theta);
    real y;
    if(K == 3){
      y = logistic3(x, theta);
    }
    if(K == 5){
      y = logistic5(x, theta);
    }
    if(K == 6){
      y = logistic6(x, theta);
    }
    return y;
  }
  
  vector nc_exp(vector mu, vector sigma, vector nu){
    /* 
    "Non-centering"" decomposition for multilevel parameters
    mu_i = mu + sigma * nu
    nu ~ normal(0, 1)
    Same as: mu_i ~ normal(mu, sigma)
    See: https://mc-stan.org/docs/2_24/stan-users-guide/reparameterization-section.html
    */
    return exp(mu + sigma .* nu);
  }
  
  vector nc_linear(vector mu, vector sigma, vector nu){
    /* 
    "Non-centering"" decomposition for multilevel parameters
    mu_i = mu + sigma * nu
    nu ~ normal(0, 1)
    Same as: mu_i ~ normal(mu, sigma)
    See: https://mc-stan.org/docs/2_24/stan-users-guide/reparameterization-section.html
    */
    return mu + sigma .* nu;
  }

}

data{
  
  int N; // total num obs
  int J; // number parameters in model A
  int K; // number parameters in model B
  int I; // number of clusters
  int id[N]; // index of cluster id
  vector[N] x;
  vector[N] y;
  
}

parameters{
  
  real Psi; // component mixture probability
  real<lower=0> sigma[2]; // component error
  vector[J] lambda_mean; // pop. mean
  vector<lower=0>[J] sigma_lambda; // pop. var
  vector[J] lambda_norm[I]; // unit normals
  vector[K] theta_mean; // pop. mean
  vector<lower=0>[K] sigma_theta; // pop. var
  vector[K] theta_norm[I]; // unit normals
  
}

transformed parameters{

  vector<lower=0>[J] lambda_i[I]; // random effect coefficients  
  vector<lower=0>[K] theta_i[I]; // random effect coefficients
  real Psi_logit = inv_logit(Psi);
  vector[N] log_lik = rep_vector(-log(N), N);

  for(i in 1:I){
    lambda_i[i] = nc_exp(lambda_mean, sigma_lambda, lambda_norm[i]);
    theta_i[i] = nc_exp(theta_mean, sigma_theta, theta_norm[i]);
  }
  
  // loop over observations
  for(n in 1:N){
    real mu1 = logisticN(x[n], lambda_i[id[n]]);
    real log_lik1 = normal_lpdf(y[n] | mu1, sigma[1]);
    real mu2 = logisticN(x[n], theta_i[id[n]]);
    real log_lik2 = normal_lpdf(y[n] | mu2, sigma[2]);
    log_lik[n] = log_mix(Psi_logit, log_lik1, log_lik2);
  }
  
}

model{
  
  for(n in 1:N){
    target += log_lik[n];
  }
  
  // priors
  Psi ~ normal(-0.6, 0.1);
  lambda_mean[1] ~ normal(log(60), 0.1);
  lambda_mean[2] ~ normal(log(50), 0.1);
  lambda_mean[3] ~ normal(log(2.5), 0.1);
  theta_mean[1] ~ normal(log(20), 0.1);
  theta_mean[2] ~ normal(log(17), 0.1);
  theta_mean[3] ~ normal(log(2), 0.1);
  theta_mean[4] ~ normal(log(85), 0.1);
  theta_mean[5] ~ normal(log(0.5), 0.1);
  theta_mean[6] ~ normal(log(0.6), 0.1);
  sigma ~ normal(0, 1);
  sigma_lambda ~ normal(0, 1);
  sigma_theta ~ normal(0, 1);
  for(i in 1:I){
    lambda_norm[i] ~ normal(0, 1);
    theta_norm[i] ~ normal(0, 1);
  }
  
}

generated quantities{
  
  vector[N] psi; // component membership probability by observation
  
  for(n in 1:N){
    real mu1 = logisticN(x[n], lambda_i[id[n]]);
    real mu2 = logisticN(x[n], theta_i[id[n]]);
    real Pr_B_given_A = exp(normal_lpdf(y[n] | mu1, sigma[1]));
    psi[n] = (Psi_logit * Pr_B_given_A) / exp(log_lik[n]);
  }
  
}
