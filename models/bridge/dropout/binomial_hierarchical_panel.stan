data {
  int<lower = 1> N;                // Total number of obs
  int<lower = 1> P;                // Total number of predictors
  int<lower=1> J;                  // number of schools
  int<lower=1> S;                  // number of students
  int<lower=1> Ti;                  // number of periods
  matrix[N, P] x;                  // predictor matrix
  int y[N];                        // Outcome of each obs
  int<lower=1,upper=J> jj[N];      // group for individual
  int<lower=1,upper=S> ss[N];      // group for student
  int<lower=1,upper=Ti> tt[N];      // group for time
}
transformed data {
  real mean_y = mean(to_vector(y));
  real<lower = 0> sd_y = sd(to_vector(y));
  //real min_y = min(to_vector(y));
  //real max_y = max(to_vector(y));
}
parameters {
  vector[P] beta;

  real<lower=0> tau;        // variance of school re
  vector[J] eta;            // standardized school re

  real<lower=0> tau_st;     // variance of student re
  vector[S] eta_st;         // standardized student re

  //real<lower=0> tau_time;     // variance of time
  //vector[Ti] eta_time;         // standardized time

}
transformed parameters {
  vector[N] x_beta;
  vector[N] alpha;
  for (n in 1:N) {
    //alpha[n] = tau * eta[jj[n]] + tau_st * eta_st[ss[n]] + tau_time * eta_time[tt[n]];    // school success log-odds
    alpha[n] = tau * eta[jj[n]] + tau_st * eta_st[ss[n]];    // school success log-odds
    x_beta[n] = x[n] * beta + alpha[n];
  }
}
model {
  target += normal_lpdf(tau | 0, 1);
  target += normal_lpdf(eta | 0, 1);
  target += normal_lpdf(tau_st | 0, 1);
  target += normal_lpdf(eta_st | 0, 1);
  //target += normal_lpdf(tau_time | 0, 1);
  //target += normal_lpdf(eta_time | 0, 1);
  // Likelihood:
  target += binomial_logit_lpmf(y | 1, x_beta);
}
generated quantities {
  vector[N] y_rep;
  vector[N] theta;
  real mean_y_rep;
  int<lower = 0, upper = 1> mean_gte;
  real sd_y_rep;
  int<lower = 0, upper = 1> sd_gte;
  //real max_y_rep;
  //int<lower = 0, upper = 1> max_gte;
  //real min_y_rep;
  //int<lower = 0, upper = 1> min_gte;

  for (n in 1:N) {
    theta[n] = inv_logit(x_beta[n]);    // school success log-odds
    y_rep[n] = binomial_rng(1, theta[n]);
  }

  mean_y_rep = mean(to_vector(y_rep));
  mean_gte = (mean_y_rep >= mean_y);
  sd_y_rep = sd(to_vector(y_rep));
  sd_gte = (sd_y_rep >= sd_y);
  // max_y_rep = max(to_vector(y_rep));
  // max_gte = (max_y_rep >= max_y);
  // min_y_rep = min(to_vector(y_rep));
  // min_gte = (min_y_rep >= min_y);
}

