data {
  int<lower = 1> N;                // Total number of obs
  int<lower = 1> P;                // Total number of predictors
  matrix[N, P] x;                  // predictor matrix
  vector[N] y;                     // Outcome of each obs
}
transformed data {
  // thin and scale the QR decomposition
  matrix[N, P] Q_ast = qr_thin_Q(x) * sqrt(N - 1);
  matrix[P, P] R_ast = qr_thin_R(x) / sqrt(N - 1);
  matrix[P, P] R_ast_inverse = inverse(R_ast);

  real mean_y = mean(to_vector(y));
  real<lower = 0> sd_y = sd(to_vector(y));
  real min_y = min(to_vector(y));
  real max_y = max(to_vector(y));
}
parameters {
  vector[P] theta;       // coefficients for predictors
  real<lower=0> sigma;  // error scale
}
model {
  // Likelihood:
  target += normal_lpdf(y | Q_ast * theta , sigma); // log-likelihood
}
generated quantities {
  vector[P] beta = R_ast_inverse * theta; // coefficients on x
  real y_rep[N] = normal_rng(Q_ast * theta, sigma);
  vector[N] res = y - Q_ast * theta;
  vector[N] res_rep = to_vector(y_rep) - Q_ast * theta;

  real mean_res = mean(to_vector(res));
  real<lower = 0> sd_res = sd(to_vector(res));
  real min_res = min(to_vector(res));
  real max_res = max(to_vector(res));

  real mean_res_rep = mean(to_vector(res_rep));
  real<lower = 0> sd_res_rep = sd(to_vector(y_rep));
  real min_res_rep = min(to_vector(res_rep));
  real max_res_rep = max(to_vector(res_rep));
  int<lower = 0, upper = 1> mean_res_gte = (mean_res_rep >= mean_res);
  int<lower = 0, upper = 1> sd_res_gte = (sd_res_rep >= sd_res);
  int<lower = 0, upper = 1> min_res_gte = (min_res_rep >= min_res);
  int<lower = 0, upper = 1> max_res_gte = (max_res_rep >= max_res);

  real mean_y_rep = mean(to_vector(y_rep));
  real<lower = 0> sd_y_rep = sd(to_vector(y_rep));
  real min_y_rep = min(to_vector(y_rep));
  real max_y_rep = max(to_vector(y_rep));
  int<lower = 0, upper = 1> mean_gte = (mean_y_rep >= mean_y);
  int<lower = 0, upper = 1> sd_gte = (sd_y_rep >= sd_y);
  int<lower = 0, upper = 1> min_gte = (min_y_rep >= min_y);
  int<lower = 0, upper = 1> max_gte = (max_y_rep >= max_y);
}
