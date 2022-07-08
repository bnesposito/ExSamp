data {
  int<lower = 1> N;                // Total number of obs
  int<lower = 1> N_obs;            // Total number of obs observed
  int<lower = 1> N_cens_U;         // Total number of obs UPPER censored
  int<lower = 1> N_cens_L;         // Total number of obs LOWER censored
  int<lower = 1> P;                // Total number of predictors
  vector[N] y;                     // Outcome of each obs
  vector[N_obs] y_obs;             // Outcome of each obs
  real<lower=max(y_obs)> U;        // Censor value (at the right)
  real<upper=min(y_obs)> L;        // Censor value (at the left)
  matrix[N, P] x;              // predictor matrix
  int<lower=1> J;                  // number of schools
  int<lower=1,upper=J> jj[N];      // group for individual
}
transformed data {
  real mean_y = mean(to_vector(y));
  real<lower = 0> sd_y = sd(to_vector(y));
}
parameters {
  vector[P] beta;
  //real<lower=0> sigma;         // population scale
  vector<lower=0>[P] sigma;         // population scale
  real<lower=0> tau;           // school level sd
  vector[J] eta;               // school level errors
}
transformed parameters {
  vector[N] x_beta;
  vector[J] alpha;
  vector[N] x_sigma;
  for (j in 1:J) {
    alpha[j] = tau * eta[j];    // school success log-odds
  }
  for (n in 1:N) {
    x_beta[n] = x[n] * beta + alpha[jj[n]];
    //x_beta[n] = x[n] * beta;
    x_sigma[n] = x[n] * sigma;
  }
}
model {
  target += normal_lpdf(tau | 0, 1);
  target += normal_lpdf(eta | 0, 1);
  target += normal_lpdf(sigma | 0, 1);
  target += normal_lpdf(beta | 0, 1);

  // Likelihood:
  for (n in 1:N){
    if (y[n] <= L) {
      //target += normal_lcdf(L | x_beta[n], sigma);
      target += normal_lcdf(L | x_beta[n], x_sigma[n]);
    } else if (y[n] >= U) {
      //target += normal_lccdf(U | x_beta[n], sigma);
      target += normal_lccdf(U | x_beta[n], x_sigma[n]);
    } else {
      //target += normal_lpdf(y[n] | x_beta[n], sigma); // log-likelihood
      target += normal_lpdf(y[n] | x_beta[n], x_sigma[n]); // log-likelihood
    }
  }
}
generated quantities {
  vector[N] y_rep;
  real mean_y_rep;
  int<lower = 0, upper = 1> mean_gte;
  real sd_y_rep;
  int<lower = 0, upper = 1> sd_gte;

  for (n in 1:N) {
    y_rep[n] = normal_rng(x_beta[n], x_sigma[n]);
    if (y_rep[n] >= U) {
      y_rep[n] = U;
    }
    if (y_rep[n] <= L) {
      y_rep[n] = L;
    }
  }

  mean_y_rep = mean(to_vector(y_rep));
  mean_gte = (mean_y_rep >= mean_y);
  sd_y_rep = sd(to_vector(y_rep));
  sd_gte = (sd_y_rep >= sd_y);
}
