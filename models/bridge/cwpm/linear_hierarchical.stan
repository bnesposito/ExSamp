data {
  int<lower = 1> N;                // Total number of obs
  int<lower = 1> P;                // Total number of predictors
  int<lower=1> J;                  // number of schools
  matrix[N, P] x;                  // predictor matrix
  real y[N];                        // Outcome of each obs
  int<lower=1,upper=J> jj[N];      // group for individual
}
transformed data {
  real mean_y = mean(to_vector(y));
  real<lower = 0> sd_y = sd(to_vector(y));
  real min_y = min(to_vector(y));
  real max_y = max(to_vector(y));
}
parameters {
  vector[P] beta;
  real<lower=0> sigma;         // population scale
  real<lower=0> tau;           // school level sd
  vector[J] eta;               // school level errors

}
transformed parameters {
  vector[N] x_beta;
  vector[J] alpha;
  for (j in 1:J) {
    alpha[j] = tau * eta[j];
  }
  for (n in 1:N) {
    x_beta[n] = x[n] * beta + alpha[jj[n]];
  }
}
model {
  target += normal_lpdf(tau | 0, 1);
  target += normal_lpdf(eta | 0, 1);
  // Likelihood:
  target += normal_lpdf(y | x_beta, sigma); // log-likelihood
}
generated quantities {
  vector[N] y_rep;
  real mean_y_rep;
  int<lower = 0, upper = 1> mean_gte;
  real sd_y_rep;
  int<lower = 0, upper = 1> sd_gte;
  real max_y_rep;
  int<lower = 0, upper = 1> max_gte;
  real min_y_rep;
  int<lower = 0, upper = 1> min_gte;

  for (n in 1:N) {
    y_rep[n] = normal_rng(x_beta[n], sigma);
  }

  mean_y_rep = mean(to_vector(y_rep));
  mean_gte = (mean_y_rep >= mean_y);
  sd_y_rep = sd(to_vector(y_rep));
  sd_gte = (sd_y_rep >= sd_y);
  max_y_rep = max(to_vector(y_rep));
  max_gte = (max_y_rep >= max_y);
  min_y_rep = min(to_vector(y_rep));
  min_gte = (min_y_rep >= min_y);
}

