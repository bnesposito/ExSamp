#TODO: test case when thompson prob(d1) = 1

testthat::test_that("Random example 1 - Sum p-shares = 1", {
  # TODO: use simulations to improve testing

  # Test multiple sample sizes
  sample_sizes = round(seq(6,1000, length.out = 5))

  for (sample_size in sample_sizes){
    outcome = sample(c(1, 0), sample_size, replace = TRUE)
    treatment = c(1, 2, 3, 4, sample(c(1, 2, 3, 4), sample_size - 4, replace = TRUE))
    alpha0 = round(runif(1, min = 1, max = 200))
    beta0 = round(runif(1, min = 1, max = 200))

    # Bayesian update for the Binomial case
    dist_cache = bayesianUpdateBinomial(treatment, outcome, alpha0, beta0)

    # Sample from the Beta posterior
    theta_matrix = sampleDistribution(dist_cache)

    # Thompson sampling
    prop_shares = thompsonSampling(theta_matrix)

    testthat::expect_equal(sum(prop_shares), 1)
  }
})
