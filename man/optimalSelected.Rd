% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/sampling.R
\name{optimalSelected}
\alias{optimalSelected}
\title{Optimal treatment selected}
\usage{
optimalSelected(shares, true_treatment_effects)
}
\arguments{
\item{shares}{A vector.}

\item{true_treatment_effects}{A vector.}
}
\value{
1 if the highest proportion treatment is the optimal treatment, 0 otherwise.
}
\description{
Find whether the optimal treatment has the highest proportion share.
}
