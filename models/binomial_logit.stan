data {
  int<lower = 1> N;                // Total number of obs
  int<lower = 1> P;                // Total number of predictors
  matrix[N, P] x;                  // predictor matrix
  int y[N];                        // Outcome of each obs
  int<lower = 1> maxT;             // Number of trials
}
parameters {
  vector[P] beta;
}
transformed parameters {
  vector[N] x_beta;
  for (n in 1:N) {
    x_beta[n] = x[n] * beta;
  }
}
model {
  // Likelihood:
  target += binomial_logit_lpmf(y | maxT, x_beta); // log-likelihood
}
generated quantities {
  vector[N] y_rep;
  vector[N] theta;

  for (n in 1:N) {
    theta[n] = inv_logit(x_beta[n]);    // school success log-odds
    y_rep[n] = binomial_rng(maxT, theta[n]);
  }
}
