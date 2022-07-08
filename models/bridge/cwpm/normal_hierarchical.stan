data {
  int<lower=0> N;              // number of observations
  int<lower=0> J;              // number of schools
  int<lower=1,upper=J> jj[N];  // group for individual
  vector[N] y;                 // outcomes
  real<lower=0> sigma;         // standard error of effect estimates
}
parameters {
  real mu;                // population treatment effect
  real<lower=0> tau;      // standard deviation in treatment effects
  vector[J] eta;          // school level errors
}
transformed parameters {
  vector[J] theta = mu + tau * eta;        // school treatment effects
}
model {
  target += normal_lpdf(eta | 0, 1);
  for (n in 1:N)
    // y[n] ~ normal(theta[jj[n]], sigma);
    target += normal_lpdf(y[n] | theta[jj[n]], sigma); // log-likelihood
}


