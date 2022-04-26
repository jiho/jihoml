#' Convert a factor into integers using 0-based indexing
#'
#' Convert a factor (or an object that can be converted as such) into an vector
#' of integers indexed from 0 (instead of 1 as usual in R). This is what xgboost
#' expects as input for classification tasks.
#'
#' @param x vector of class factor (or character).
#'
#' @return A vector of integers encoding factors, starting at 0.

#' @export
#' @examples
#' defactor(factor(c("a", "b", "a")))
defactor <- function(x) {
  if (!inherits(x, "factor")) {
    warning("Argument is not a factor; converting it")
    x <- factor(x)
  }
  xd <- as.integer(x) - 1L
  attr(xd, "levels") <- levels(x)
  return(xd)
}

#' Convert a vector of 0-based indexed integers into a factor
#'
#' Convert a 0-based indexed vector, which is the output of xgboost for
#' classification, into an R factor.
#'
#' @param x vector of integers representing a factor, with 0-based indexing.
#' @param levels levels of the factor to recreate, given in order.
#'
#' @return A vector of class factor

#' @export
#' @examples
#' xf <- factor(c("a", "b", "a"))
#' xi <- defactor(xf)
#' xi
#' xf2 <- refactor(xi, levels=c("a", "b"))
#' identical(xf, xf2)
refactor <- function(x, levels) {
  if (is.numeric(x)) {
    if (!is.integer(x)) {
      warning("Argument is not integer; converting it")
      x <- as.integer(x)
    }
  } else {
    stop("Argument needs to be integer (or at least numeric)")
  }
  xf <- factor(levels[x+1L], levels=levels)
  return(xf)
}
