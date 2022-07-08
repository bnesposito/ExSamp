data {
  int<lower = 1> N;                // Total number of obs
  int<lower = 1> P;                // Total number of predictors
  int<lower=1> J;                  // number of schools
  matrix[N, P] x;                  // predictor matrix
  int y[N];                        // Outcome of each obs
  int<lower=1,upper=J> jj[N];      // group for individual
  int<lower = 1> maxT;             // Number of trials
}
parameters {
  vector[P] beta;
  //real mu;                // population mean of success log-odds
  real<lower=0> tau;      // population sd of success log-odds
  vector[J] eta;               // school level errors of success log-odds
}
transformed parameters {
  vector[N] x_full;
  vector[J] alpha;
  for (j in 1:J) {
    alpha[j] = tau * eta[j];    // school success log-odds
  }
  for (n in 1:N) {
    x_full[n] = x[n] * beta + alpha[jj[n]];
  }
}
model {
  target += normal_lpdf(tau | 0, 1);
  target += normal_lpdf(eta | 0, 1);
  target += binomial_logit_lpmf(y | maxT, x_full); // log-likelihood
}

