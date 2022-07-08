data {
  int<lower=1> N;              // number of observations
  int<lower=1> J;              // number of schools
  int<lower=1> K;              // number of treatments
  int<lower=1,upper=J> jj[N];  // group for individual
  int<lower=1,upper=K> kk[N];  // treatment for individual
  vector<lower=0,upper=85>[N] y;                 // outcomes
}
transformed data {
  real mean_y = mean(to_vector(y));
  real<lower = 0> sd_y = sd(to_vector(y));
  real min_y = min(to_vector(y));
  real max_y = max(to_vector(y));
}
parameters {
  vector<lower=0,upper=85>[K] mu;                // population treatment effect
  vector<lower=0>[K] tau;      // standard deviation in treatment effects
  vector[K*J] eta;             // school level error
  vector<lower=0>[K*J] sigma;    // standard error of outcomess
}
transformed parameters {
  vector[K*J] theta;
  real mean_theta;
  real<lower = 0> sd_theta;
  real min_theta;
  real max_theta;

  for (k in 1:K) {
    theta[(k-1)*J+1:k*J] = mu[k] + tau[k] * eta[(k-1)*J+1:k*J];    // school treatment effects
  }

  mean_theta = mean(to_vector(theta));
  sd_theta = sd(to_vector(theta));
  min_theta = min(to_vector(theta));
  max_theta = max(to_vector(theta));
}
model {
  target += inv_chi_square_lpdf(sigma | 2);
  target += normal_lpdf(eta | 0, 1);
  for (n in 1:N)
    target += normal_lpdf(y[n] | theta[(kk[n]-1)*J+jj[n]], sigma[(kk[n]-1)*J+jj[n]]); // log-likelihood
}
generated quantities {
  // theta replicated;  mu and tau not replicated
  real theta_rep[K*J];
  real y_rep[N];

  real mean_y_rep;
  real<lower = 0> sd_y_rep;
  real min_y_rep;
  real max_y_rep;

  int<lower = 0, upper = 1> mean_gte;
  int<lower = 0, upper = 1> sd_gte;
  int<lower = 0, upper = 1> min_gte;
  int<lower = 0, upper = 1> max_gte;

  real mean_theta_rep;
  real<lower = 0> sd_theta_rep;
  real min_theta_rep;
  real max_theta_rep;

  int<lower = 0, upper = 1> mean_gte_theta;
  int<lower = 0, upper = 1> sd_gte_theta;
  int<lower = 0, upper = 1> min_gte_theta;
  int<lower = 0, upper = 1> max_gte_theta;

  for (k in 1:K) {
    for (j in 1:J) {
      theta_rep[(k-1)*J+j] = normal_rng(mu[k], tau[k]);
    }
  }
  for (n in 1: N) {
    y_rep[n] = normal_rng(theta_rep[(kk[n]-1)*J+jj[n]], sigma[(kk[n]-1)*J+jj[n]]);
  }

  mean_y_rep = mean(to_vector(y_rep));
  sd_y_rep = sd(to_vector(y_rep));
  min_y_rep = min(to_vector(y_rep));
  max_y_rep = max(to_vector(y_rep));

  mean_gte = (mean_y_rep >= mean_y);
  sd_gte = (sd_y_rep >= sd_y);
  min_gte = (min_y_rep >= min_y);
  max_gte = (max_y_rep >= max_y);

  mean_theta_rep = mean(to_vector(theta_rep));
  sd_theta_rep = sd(to_vector(theta_rep));
  min_theta_rep = min(to_vector(theta_rep));
  max_theta_rep = max(to_vector(theta_rep));

  mean_gte_theta = (mean_theta_rep >= mean_theta);
  sd_gte_theta = (sd_theta_rep >= sd_theta);
  min_gte_theta = (min_theta_rep >= min_theta);
  max_gte_theta = (max_theta_rep >= max_theta);
}
