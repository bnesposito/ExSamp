data {
  int<lower = 1> N;                // Total number of obs
  int<lower = 1> K;                // Total number of treatments
  vector[N] y;                     // Outcome of each obs
  int<lower=1,upper=K> kk[N];      // Treatment of each
}
transformed data {
  real mean_y = mean(to_vector(y));
  real<lower = 0> sd_y = sd(to_vector(y));
  real min_y = min(to_vector(y));
  real max_y = max(to_vector(y));
}
parameters {
  vector[K] mu;
  vector<lower = 0>[K] sigma;
}
model {
  // Likelihood:
  for (n in 1:N)
    target += normal_lpdf(y[n] | mu[kk[n]], sigma[kk[n]]); // log-likelihood
}
generated quantities {
  real y_rep[N] = normal_rng(mu[kk], sigma[kk]);
  real mean_y_rep = mean(to_vector(y_rep));
  real<lower = 0> sd_y_rep = sd(to_vector(y_rep));
  real min_y_rep = min(to_vector(y_rep));
  real max_y_rep = max(to_vector(y_rep));
  int<lower = 0, upper = 1> mean_gte = (mean_y_rep >= mean_y);
  int<lower = 0, upper = 1> sd_gte = (sd_y_rep >= sd_y);
  int<lower = 0, upper = 1> min_gte = (min_y_rep >= min_y);
  int<lower = 0, upper = 1> max_gte = (max_y_rep >= max_y);
}
