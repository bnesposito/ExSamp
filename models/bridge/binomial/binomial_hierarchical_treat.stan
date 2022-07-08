data {
  int<lower=1> N;              // number of observations
  int<lower=1> J;              // number of schools
  int<lower=1> K;              // number of treatments
  int<lower=1,upper=J> jj[N];  // group for individual
  int<lower=1,upper=K> kk[N];  // treatment for individual
  int y[N];                 // outcomes
  int<lower = 1> maxT;             // Number of trials
}
transformed data {
  real mean_y = mean(to_vector(y));
  //real<lower = 0> sd_y = sd(to_vector(y));
  //real min_y = min(to_vector(y));
  //real max_y = max(to_vector(y));
}
parameters {
  vector[K] mu;                // population mean of success log-odds
  vector<lower=0>[K] tau;      // population sd of success log-odds
  vector[K*J] eta;             // school level errors of success log-odds
}
transformed parameters {
  vector[K*J] alpha;
  for (k in 1:K) {
    alpha[(k-1)*J+1:k*J] = mu[k] + tau[k] * eta[(k-1)*J+1:k*J];    // school success log-odds
  }
}
model {
  //target += normal_lpdf(mu | -1, 1);
  target += normal_lpdf(tau | 0, 1);
  target += normal_lpdf(eta | 0, 1);
  for (n in 1:N)
    target += binomial_logit_lpmf(y[n] | maxT, alpha[(kk[n]-1)*J+jj[n]]); // log-likelihood
}
generated quantities {
  vector[K*J] theta;  // chance of success
  vector[K] mu_prob;  // chance of success
  vector[N] y_rep;
  real mean_y_rep;
  int<lower = 0, upper = 1> mean_gte;

  for (kj in 1:K*J) {
    theta[kj] = inv_logit(alpha[kj]);    // school success log-odds
  }
  for (k in 1:K) {
    mu_prob[k] = inv_logit(mu[k]);    // school success log-odds
  }
  for (n in 1:N) {
    y_rep[n] = binomial_rng(maxT, theta[(kk[n]-1)*J+jj[n]]);
  }
  mean_y_rep = mean(to_vector(y_rep));
  mean_gte = (mean_y_rep >= mean_y);

  // theta replicated;  mu and tau not replicated
  //real theta_rep[K*J];
  //real y_rep[N];
  //real mean_y_rep;
  //real<lower = 0> sd_y_rep;
  //real min_y_rep;
  //real max_y_rep;
  //int<lower = 0, upper = 1> mean_gte;
  //int<lower = 0, upper = 1> sd_gte;
  //int<lower = 0, upper = 1> min_gte;
  //int<lower = 0, upper = 1> max_gte;

  //for (k in 1:K) {
  //  for (j in 1:J) {
  //    theta_rep[(k-1)*J+j] = normal_rng(mu[k], tau[k]);
  //  }
  //}
  //for (n in 1: N) {
  //  y_rep[n] = normal_rng(theta_rep[(kk[n]-1)*J+jj[n]], sigma);
  //}

  //mean_y_rep = mean(to_vector(y_rep));
  //sd_y_rep = sd(to_vector(y_rep));
  //min_y_rep = min(to_vector(y_rep));
  //max_y_rep = max(to_vector(y_rep));
  //mean_gte = (mean_y_rep >= mean_y);
  //sd_gte = (sd_y_rep >= sd_y);
  //min_gte = (min_y_rep >= min_y);
  //max_gte = (max_y_rep >= max_y);
}

