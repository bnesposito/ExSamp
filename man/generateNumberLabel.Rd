% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/consilium.R
\name{generateNumberLabel}
\alias{generateNumberLabel}
\title{Phone number label generator}
\usage{
generateNumberLabel(wave_allocation, N_MESSAGES_LIMIT)
}
\arguments{
\item{wave_allocation}{A vector with the treatment allocation for each observation.}

\item{N_MESSAGES_LIMIT}{A scalar. The maximum number of messages per phone number.}
}
\value{
A vector with the phone number labels for each observation.
}
\description{
Generates a phone number label to be used by consilium. Each number is limited to N_MESSAGES_LIMIT per number.
}
