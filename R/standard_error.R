#' Compute the standard error of the mean, assuming a normal distribution
#'
#' @param x numeric vector.
#'
#' @returns The value of the standard error of the mean = sd / sqrt(n)
#' @export
se <- function(x) {stats::sd(x)/sqrt(length(x))}
