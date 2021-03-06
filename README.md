exSamp: Exploration Sampling
================

<!-- badges: start -->

[![R-CMD-check](https://github.com/Exploration-Sampling/exSamp/workflows/R-CMD-check/badge.svg)](https://github.com/Exploration-Sampling/exSamp/actions)
[![packageversion](https://img.shields.io/badge/Package%20version-0.0.1.1000-orange.svg?style=flat-square)](commits/master)
[![minimal R
version](https://img.shields.io/badge/R%3E%3D-4.0.3-6666ff.svg)](https://cran.r-project.org/)
[![Last-changedate](https://img.shields.io/badge/last%20change-2021--03--03-yellowgreen.svg)](/commits/master)
<!-- badges: end -->

The R code in this repository provides an application that can carry out policy choice experiments using the 
Exploration Sampling algorithm (Kasy & Sautmann, 2021). The code can accommodate a range of outcome distributions 
and models. Currently implemented are:
  - The Beta Bernoulli model used in Kasy & Sautmann, 2021.
  - A hierarchical Binomial model as in Esposito & Sautmann, 2022.
  - A simple Binomial and a Poisson model.

For details see Esposito & Sautmann [working paper](https://drive.google.com/file/d/1HJCZQzavbKgGE3n8w4TiGviMq7wi37Vf/view)
The simple binomial and Bernoulli models are also available as a simple to use and intuitive [dashboard](https://bnesposito.shinyapps.io/exsampdashboard/).

## Structure and folders

  - **R**: contains the functions of the package. All functions placed
    here will be imported when using “devtools::load\_all()”. All functions related to 
    modelling and computation are in the folder Sampling. All functions related to plots are 
    in the folder Ploting.
  - **man**: documentation of the package. This is automatized through
    the package roxygen2.
  - **tests**: contains the tests of the package. Each R file is
    associated to a function in the package.
  - **models**: contains multiple STAN models.
  - **docs**: additional documentation of the package manually
    generated.
  - **data**: contains data to be used in the examples of the package.

### Application flowchart

![Main
flowchart](https://github.com/bnesposito/exSamp/blob/master/docs/main_flowchart_v09.svg?raw=true)

## Testing status (74.1% test coverage)

> Test coverage is defined as a metric in Software Testing that measures
> the amount of testing performed by a set of tests. It will include
> gathering information about which parts of a program are executed when
> running the test suite to determine which branches of conditional
> statements have been taken into account.

  - **bayesianUpdateBinomial** (Total tests: 20)
      - Scope 1 - Specific example:
          - Prior: uniform
          - Treatments: 3
          - Sample size: 6
      - Scope 2 - Random example:
          - Prior: random beta
          - Treatments: 6
          - Sample size: random (3 times)
      - Subject:
          - Output size.
          - Correct computation of posterior parameters.
          - Parameters greater than 0.
  - **sampleDistribution** (Total tests: 72)
      - Scope 1 - Random example:
          - Sample resolution: 100k
          - Prior: random beta
          - Treatments: 4
          - Sample size: random (3 times)
      - Subject:
          - Mean from the sample draws is close to the theoretical mean
            of the posterior distribution with 0.001 tolerance.
          - Variance from the sample draws is close to the theoretical
            variance of the posterior distribution with 0.001 tolerance.
          - Skewness from the sample draws is close to the theoretical
            skewness of the posterior distribution with 0.05 tolerance.
  - **thompsonSampling** (Total tests: 5)
      - Scope 1 - Random example:
          - Prior: random beta
          - Treatments: 4
          - Sample size: random (5 times)
      - Subject:
          - Check that the sum of the p-shares is equal to 1.
          - (TODO) Test case when Thompson p-share(d1) = 1
          - (TODO) Check asymptotic behavior
  - **explorationSampling** (Total tests: 5)
      - Scope 1 - Random example:
          - Prior: random beta
          - Treatments: 4
          - Sample size: random (5 times)
      - Subject:
          - Check that the sum of the p-shares is equal to 1.
          - (TODO) Test case when exploration p-share(d1) = 1
          - (TODO) Check asymptotic behavior
  - **proportionalAssignment** (Total tests: 18)
      - Scope 1 - Random example:
          - Prior: random beta
          - Treatments: 3
          - Sample size: random (3 times)
          - Simulation runs: 1k
      - Subject:
          - Check that the empirically asymptotic p-shares are close to
            the true treatment p-share with a tolerance of 0.01.
