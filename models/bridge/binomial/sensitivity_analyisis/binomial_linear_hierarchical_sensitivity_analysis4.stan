// Sensitivity analysis 4
// We add an specific prioir to the vector of betas:
//   target += normal_lpdf(beta | 0, 100);

data {
  int<lower = 1> N;                // Total number of obs
  int<lower = 1> P;                // Total number of predictors
  int<lower=1> J;                  // number of schools
  matrix[N, P] x;                  // predictor matrix
  int y[N];                        // Outcome of each obs
  int<lower=1,upper=J> jj[N];      // group for individual
  int<lower = 1> maxT;             // Number of trials
}
transformed data {
  real mean_y = mean(to_vector(y));
  real<lower = 0> sd_y = sd(to_vector(y));
  //real min_y = min(to_vector(y));
  //real max_y = max(to_vector(y));
}
parameters {
  vector[P] beta;
  //real mu;                // population mean of success log-odds
  real<lower=0> tau;      // population sd of success log-odds
  vector[J] eta;               // school level errors of success log-odds
}
transformed parameters {
  vector[N] x_beta;
  vector[J] alpha;
  for (j in 1:J) {
    alpha[j] = tau * eta[j];    // school success log-odds
  }
  for (n in 1:N) {
    x_beta[n] = x[n] * beta + alpha[jj[n]];
  }
}
model {
  target += normal_lpdf(beta | 0, 100);
  target += normal_lpdf(tau | 0, 1);
  target += normal_lpdf(eta | 0, 1);
  // Likelihood:
  target += binomial_logit_lpmf(y | maxT, x_beta); // log-likelihood
}
generated quantities {
  vector[N] y_rep;
  vector[N] theta;
  real mean_y_rep;
  int<lower = 0, upper = 1> mean_gte;
  real sd_y_rep;
  int<lower = 0, upper = 1> sd_gte;

  for (n in 1:N) {
    theta[n] = inv_logit(x_beta[n]);    // school success log-odds
    y_rep[n] = binomial_rng(maxT, theta[n]);
  }

  mean_y_rep = mean(to_vector(y_rep));
  mean_gte = (mean_y_rep >= mean_y);
  sd_y_rep = sd(to_vector(y_rep));
  sd_gte = (sd_y_rep >= sd_y);
}
