functions{
  
  real logistic3(real x, vector theta){
    // 3 parameter logistic function
    // theta[1] is the lower y-asymptote
    // theta[2] is the inflection point (midpoint) of the curve (x-axis units)
    // theta[3] controls the stepness of the curve
    real y;
    y = theta[1] / (1+exp((theta[2]-x)/theta[3]));
    return y;
  }
  
  real logistic5(real x, vector theta){
    // 5 parameter logistic function
    // theta[1] is the upper y-asymptote
    // theta[2] is the x-coordinate of the first inflection point
    // theta[3] is the first growth rate parameter
    // theta[4] is the x-coordinate of the second inflection point
    // theta[5] is the second growth rate parameter
    real y;
    y = theta[1] / (1+exp((theta[2]-x)/theta[3])) - 
        theta[1] / (1+exp((theta[4]-x)/theta[5]));
    return y;
  }
  
  real logistic6(real x, vector theta){
    // 6 parameter logistic function
    // theta[1] is the upper y-asymptote
    // theta[2] is the x-coordinate of the first inflection point
    // theta[3] is the first growth rate parameter
    // theta[4] is the x-coordinate of the second inflection point
    // theta[5] is the second growth rate parameter
    // theta[6] is related to the value of the second lower asymptote
    real y;
    y = theta[1] / (1+exp((theta[2]-x)/theta[3])) - 
        (theta[1]*theta[6]) / (1+exp((theta[4]-x)/theta[5]));
    return y;
  }

  real logisticN(real x, vector theta){
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
    // "Non-centering"" decomposition for multilevel parameters
    // mu_i = mu + sigma * nu
    // nu ~ normal(0, 1)
    // Same as: mu_i ~ normal(mu, sigma)
    return exp(mu + sigma .* nu);
  }
  
  vector nc_linear(vector mu, vector sigma, vector nu){
    // "Non-centering"" decomposition for multilevel parameters
    // mu_i = mu + sigma * nu
    // nu ~ normal(0, 1)
    // Same as: mu_i ~ normal(mu, sigma)
    return mu + sigma .* nu;
  }

}

data{
  
  int N; // total num obs
  int K; // number parameters
  int I; // number of clusters
  int id[N]; //index of cluster id
  vector[N] x;
  vector[N] y;
  
}

parameters{
  
  real<lower=0> sigma; // error
  vector[K] theta_mean; // pop. mean
  vector<lower=0>[K] sigma_theta; // pop. var
  vector[K] theta_norm[I]; // unit normals

}

transformed parameters{
  
  vector<lower=0>[K] theta_i[I]; // random effect coefficients
  
  for(i in 1:I){
    theta_i[i] = nc_exp(theta_mean, sigma_theta, theta_norm[i]);
  }

}

model{
  
  // loop over observations
  for(n in 1:N){
    real mu = logisticN(x[n], theta_i[id[n]]);
    target += normal_lpdf(y[n] | mu, sigma); // increment log prob
  }
    
  // priors
  theta_mean ~ normal(0, 1);
  sigma ~ normal(0, 1);
  sigma_theta ~ normal(0, 0.1);
  for(i in 1:I){
    theta_norm[i] ~ normal(0, 1);
  }
  
}

generated quantities{

  vector[N] y_hat; // predicted values
  vector[N] log_lik; // log-likelihood

  for(n in 1:N){
    real mu;
    mu = logisticN(x[n], theta_i[id[n]]);
    y_hat[n] = normal_rng(mu, sigma);
    log_lik[n] = normal_lpdf(y[n] | logisticN(x[n], theta_i[id[n]]), sigma);
  }

}
