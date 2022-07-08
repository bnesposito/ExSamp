testthat::test_that("Random example 1 - Big sample", {
  # Check that the empirically asymptotic p-shares are close to the true treatment p-share with a tolerance of 0.005
  # TODO: random p-shares

  tolerance = 5e-3
  n_repetitions = 1000
  p_shares = c(0.2,0.3,0.5)

  # Test multiple sample sizes
  sample_sizes = round(seq(50,1000, length.out = 4))

  for (sample_size in sample_sizes){
    new_sample = proportionalAssignment(p_shares, sample_size)
    treatment_count = table(new_sample)

    for (i in 1:n_repetitions) {
        new_sample = proportionalAssignment(p_shares, sample_size)
        treatment_count = treatment_count + table(new_sample)
    }

    final_p_shares = treatment_count/sum(treatment_count)

    for (i in 1:length(p_shares)){
      testthat::expect_lt(final_p_shares[i] - p_shares[i], tolerance)
      testthat::expect_gt(final_p_shares[i] - p_shares[i], -tolerance)
    }
  }
})

testthat::test_that("Random example 2 - Small sample", {
  # Check that the empirically asymptotic p-shares are close to the true treatment p-share with a tolerance of 0.006
  # TODO: random p-shares

  tolerance = 6e-3
  n_repetitions = 1000
  p_shares = c(0.2,0.3,0.5)

  # Test multiple sample sizes
  sample_sizes = round(seq(7,21, length.out = 4))

  for (sample_size in sample_sizes){
    new_sample = proportionalAssignment(p_shares, sample_size)
    treatment_count = table(new_sample)

    for (i in 1:n_repetitions) {
      new_sample = proportionalAssignment(p_shares, sample_size)
      treatment_count = treatment_count + table(new_sample)
    }

    final_p_shares = treatment_count/sum(treatment_count)

    for (i in 1:length(p_shares)){
      testthat::expect_lt(final_p_shares[i] - p_shares[i], tolerance)
      testthat::expect_gt(final_p_shares[i] - p_shares[i], -tolerance)
    }
  }
})


testthat::test_that("Random example 2 - Small sample, small treatment p-share", {
  # Check that the empirically asymptotic p-shares are close to the true treatment p-share with a tolerance of 0.006
  # TODO: random p-shares

  tolerance = 6e-3
  n_repetitions = 1000
  p_shares = c(0.05,0.45,0.5)

  # Test multiple sample sizes
  sample_sizes = round(seq(7,21, length.out = 4))

  for (sample_size in sample_sizes){
    new_sample = proportionalAssignment(p_shares, sample_size)
    treatment_count = table(factor(new_sample, 1:length(p_shares)))

    for (i in 1:n_repetitions) {
      new_sample = proportionalAssignment(p_shares, sample_size)
      treatment_count = treatment_count + table(factor(new_sample, 1:length(p_shares)))
    }

    final_p_shares = treatment_count/sum(treatment_count)

    for (i in 1:length(p_shares)){
      testthat::expect_lt(final_p_shares[i] - p_shares[i], tolerance)
      testthat::expect_gt(final_p_shares[i] - p_shares[i], -tolerance)
    }
  }
})
