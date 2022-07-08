data {
  int<lower = 1> N;                // Total number of obs
  int<lower = 1> N_obs;            // Total number of obs observed
  int<lower = 1> N_cens_U;         // Total number of obs UPPER censored
  int<lower = 1> N_cens_L;         // Total number of obs LOWER censored
  int<lower = 1> K;                // Total number of treatments
  vector[N] y;                     // Outcome of each obs
  vector[N_obs] y_obs;             // Outcome of each obs
  real<lower=max(y_obs)> U;        // Censor value (at the right)
  real<upper=min(y_obs)> L;        // Censor value (at the left)
  int<lower=1,upper=K> kk[N_obs];  // Treatment of each obs observed
  int<lower=1,upper=K> kk_cens_U[N_cens_U];  // Treatment of each obs censored
  int<lower=1,upper=K> kk_cens_L[N_cens_L];  // Treatment of each obs censored
}
transformed data {
  real mean_y = mean(to_vector(y));
  real<lower = 0> sd_y = sd(to_vector(y));
}
parameters {
  real<lower = 0, upper = 85> mu;
  real<lower = 0> sigma;
}
model {
  // Likelihood:
  target += normal_lpdf(sigma | 0, 100);
  target += normal_lpdf(y_obs | mu, sigma); // log-likelihood
  target += N_cens_U * normal_lccdf(U | mu, sigma);
  target += N_cens_L * normal_lcdf(L | mu, sigma);
}
generated quantities {
  vector[N] y_rep;
  real mean_y_rep;
  int<lower = 0, upper = 1> mean_gte;
  real sd_y_rep;
  int<lower = 0, upper = 1> sd_gte;

  for (n in 1:N) {
    y_rep[n] = normal_rng(mu, sigma);
    if (y_rep[n] > U) {
      y_rep[n] = U;
    }
    if (y_rep[n] < L) {
      y_rep[n] = L;
    }
  }

  mean_y_rep = mean(to_vector(y_rep));
  mean_gte = (mean_y_rep >= mean_y);
  sd_y_rep = sd(to_vector(y_rep));
  sd_gte = (sd_y_rep >= sd_y);
}

