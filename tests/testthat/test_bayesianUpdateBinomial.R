# To run tests:
# testthat::test_dir("tests/testthat")
# devtools::test("tests")
# devtools::check()
# covr::package_coverage()

# devtools::load_all()

testthat::test_that("Specific example 1 - Test output size, correct computation", {
  outcome = c(1, 1, 1, 0, 0, 0)
  treatment = c(1, 1, 2, 2, 5, 5)

  expected_alpha = as.array(c(3, 2, 1))
  names(expected_alpha) = c('treat1', 'treat2', 'treat5')

  expected_beta = as.array(c(1, 2, 3))
  names(expected_beta) = c('treat1', 'treat2', 'treat5')

  # Check correct treatment size
  testthat::expect_equal(bayesianUpdateBinomial(treatment, outcome, alpha0 = 1, beta0 = 1)$n_treatment, 3)

  # Check correct computation of posterior parameters
  testthat::expect_equal(bayesianUpdateBinomial(treatment, outcome, alpha0 = 1, beta0 = 1)$parameters$alpha, expected_alpha)
  testthat::expect_equal(bayesianUpdateBinomial(treatment, outcome, alpha0 = 1, beta0 = 1)$parameters$beta, expected_beta)

  # Check correct posterior parameters size
  testthat::expect_length(bayesianUpdateBinomial(treatment, outcome, alpha0 = 1, beta0 = 1)$parameters$alpha, 3)
  testthat::expect_length(bayesianUpdateBinomial(treatment, outcome, alpha0 = 1, beta0 = 1)$parameters$beta, 3)
})

testthat::test_that("Random example 1 - Test output size, correct computation and limits (>0)", {

  # Test multiple sample sizes
  sample_sizes = round(seq(6,1000, length.out = 3))

  for (sample_size in sample_sizes){
    outcome = sample(c(1, 0), sample_size, replace = TRUE)
    treatment = c(1,2,3,4,5,6, sample(c(1, 2, 3, 4, 5, 6), sample_size - 6, replace = TRUE))
    alpha0 = round(runif(1, min = 1, max = 200))
    beta0 = round(runif(1, min = 1, max = 200))

    N = table(treatment)
    S = tapply(outcome, treatment, sum, default = 0)

    expected_alpha = as.array(c(alpha0 + S))
    names(expected_alpha) = c('treat1', 'treat2', 'treat3', 'treat4', 'treat5', 'treat6')

    expected_beta = as.array(c(beta0 + N - S))
    names(expected_beta) = c('treat1', 'treat2', 'treat3', 'treat4', 'treat5', 'treat6')

    # Check correct treatment size
    testthat::expect_equal(bayesianUpdateBinomial(treatment, outcome, alpha0, beta0)$n_treatment, 6)

    # Check correct computation of posterior parameters
    testthat::expect_equal(bayesianUpdateBinomial(treatment, outcome, alpha0, beta0)$parameters$alpha, expected_alpha)
    testthat::expect_equal(bayesianUpdateBinomial(treatment, outcome, alpha0, beta0)$parameters$beta, expected_beta)

    # Check that posterior parameters are not smaller than 0
    testthat::expect_equal(sum(bayesianUpdateBinomial(treatment, outcome, alpha0, beta0)$parameters$alpha < 0), 0)
    testthat::expect_equal(sum(bayesianUpdateBinomial(treatment, outcome, alpha0, beta0)$parameters$beta < 0), 0)
  }
})

