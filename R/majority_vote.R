#' Return the majority vote in a discrete valued vector
#'
#' @param x a vector coercible to a factor.
#'
#' @returns The most common value in x, as a factor.
#' @export
#' @examples
#' majority_vote(c(1,2,3,3,3))
#' majority_vote(factor(mtcars$cyl))
majority_vote <- function(x) {
  if (!is.factor(x)) {
    warning("Converting to factor")
    x <- factor(x)
  }
  imax <- which.max(tabulate(x))
  return(factor(levels(x)[imax], levels=levels(x)))
}
