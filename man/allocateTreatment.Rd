% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/consilium.R
\name{allocateTreatment}
\alias{allocateTreatment}
\title{Allocate treatment}
\usage{
allocateTreatment(
  initial_data,
  wave_shares,
  WAVE_SIZE,
  WAVE_INDEX,
  N_MESSAGES_LIMIT
)
}
\arguments{
\item{initial_data}{A data.frame containing (i) ids, (ii) treatment, (iii) wave and (iv) phone_number. Observations with
missing values in the variable treatment are available for treatment.}

\item{wave_shares}{A vector with the proportional shares for each treatment.}

\item{WAVE_SIZE}{A scalar. The number of observations to be allocated.}

\item{WAVE_INDEX}{A scalar. Index of the wave.}

\item{N_MESSAGES_LIMIT}{A scalar. The maximum number of messages per phone number.}
}
\value{
A data.frame with the allocated treatments.
}
\description{
Allocate treatment based on the proportions on the vector wave_shares. Then, randomly fill the rows of the dataframe
}
