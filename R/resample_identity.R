#' Generate repeated resamples of the same data
#'
#' This is a dummy function, used to check the consistency of further computation.
#' It just reshuffles the rows of the input data.
#'
#' @inheritParams resample_boot
#' @param n integer, number of repetitions.
#'
#' @return A tibble with columns
#' - train : an object of class modelr::resample. The input data with reshuffled rows.
#' - val   : an empty object of class modelr::resample.
#' - id    : integer, the repetition number.
#'
#' @export
#' @examples
#' rs <- resample_identity(mtcars, n=3)
#' rs
#' data.frame(rs$train[1])
#' data.frame(rs$train[2])
#' # = same except for the row order
resample_identity <- function(data, n=10) {
  if (n < 0) stop("The number of repetitions should be > 0.")

  # convert input data to data.frame for modelr::resample
  data_df <- as.data.frame(data)

  # precompute some commonly used variables
  nr <- nrow(data)

  ids <- purrr::map_dfr(1:n, function(i){
    dplyr::tibble(
      # reshuffle rows
      train = list(modelr::resample(data=data_df, idx=sample.int(nr))),
      # empty
      val   = list(modelr::resample(data=data_df, idx=vector(mode="integer", length=0))),
      id=i
    )
  })

  # add a special class for further processing
  class(ids) <- c("resamples", class(ids))

  return(ids)
}
