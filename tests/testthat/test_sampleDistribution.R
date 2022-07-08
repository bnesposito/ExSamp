testthat::test_that("Random example 1 - Binomial case - Test mean, variance, skewness close to theoretical values", {
  # Test that the mean and variance drawn from the posterior are close to the theoretical mean and variance of the posterior distribution.

  # Tolerance value
  epsilon = 1e-3
  epsilon_skewness = 5e-2

  # Test multiple sample sizes
  sample_sizes = round(seq(6,1000, length.out = 3))

  for (sample_size in sample_sizes){
    outcome = sample(c(1, 0), sample_size, replace = TRUE)
    treatment = c(1, 2, 3, 4, sample(c(1, 2, 3, 4), sample_size - 4, replace = TRUE))
    alpha0 = round(runif(1, min = 1, max = 200))
    beta0 = round(runif(1, min = 1, max = 200))

    dist_cache = bayesianUpdateBinomial(treatment, outcome, alpha0, beta0)
    theta_matrix = sampleDistribution(dist_cache)

    for (i in 1:dist_cache$n_treatment) {
      # Get mean and variance from the sample
      theta_sample = theta_matrix[i,]
      theta_mean = mean(theta_sample)
      theta_var = sd(theta_sample)^2
      theta_skewness = moments::skewness(theta_sample)

      # Get theoretical mean and variance
      alpha = dist_cache$parameters$alpha[i]
      beta = dist_cache$parameters$beta[i]

      theory_mean = alpha/(alpha+beta)
      theory_var = alpha*beta/((alpha+beta)^2 * (alpha+beta+1))
      theory_skewness = (2*(beta-alpha)*sqrt(alpha+beta+1))/((alpha+beta+2)*sqrt(alpha*beta))

      # Test that the mean is close to the theoretical mean
      testthat::expect_gt(theta_mean, theory_mean - epsilon)
      testthat::expect_lt(theta_mean, theory_mean + epsilon)

      # Test that the var is close to the theoretical mean
      testthat::expect_gt(theta_var, theory_var - epsilon)
      testthat::expect_lt(theta_var, theory_var + epsilon)

      # Test that the skewness is close to the theoretical mean
      testthat::expect_gt(theta_skewness, theory_skewness - epsilon_skewness)
      testthat::expect_lt(theta_skewness, theory_skewness + epsilon_skewness)
    }
  }
})
