#' Replicate each row of a resamples object
#'
#' This is useful to test the replicability of a modelling pipeline or to prepare for permutations.
#'
#' @param object of class resamples, created by a `resample_***()` function.
#' @param n number of times to replicate
#'
#' @examples
#' rs <- resample_split(mtcars, p=0.7)
#' replicate(rs, n=5)
#' rs <- resample_cv(mtcars, k=3, n=2)
#' replicate(rs, n=3)
replicate <- function(object, n, ...) {
  # replicate the rows
  n_resamples <- nrow(object)
  x <- object[rep(1:n_resamples, each=n),]
  # identifiy the replicate
  x$replic <- rep(1:n, times=n_resamples)
  return(x)
}
