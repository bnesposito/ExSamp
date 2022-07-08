data {
  int<lower = 1> N;  // Total number of obs
  vector[N] y;       // Outcome of each obs
}
parameters {
  real mu;
  real<lower = 0> sigma;
}
model {
  // Priors:
  target += normal_lpdf(mu | 0, 200);
  target += lognormal_lpdf(sigma | 3, 1);
  // Likelihood:
  target += normal_lpdf(y | mu, sigma);
}
