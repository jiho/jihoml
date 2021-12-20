#' Generate train-val splits of the data
#'
#' @inheritParams resample_boot
#' @param p in \[0,1\], the proportion of observations to use for training.
#' @param n integer, number of repetitions of the split.
#'
#' @return A tibble with columns
#' - train : an object of class modelr::resample. The training data.
#' - val   : an object of class modelr::resample. The validation data (i.e. the
#'           rows of .data not selected in the training data).
#' - repet : integer, the repetition number.
#'
#' @export
#' @examples
#' resample_split(mtcars, p=0.7, n=5)
#'
#' # stratify train-val by gear
#' rs  <- resample_split(mtcars, p=0.5, n=10)
#' rss <- resample_split(mtcars, p=0.5, n=10, gear)
#' sapply(rs$train, function(x) {sum(data.frame(x)$gear==4)})
#' # = variable number of occurrence of gear==4 in the training set
#' sapply(rss$train, function(x) {sum(data.frame(x)$gear==4)})
#' # = reliable number of gear==4 in the training set
resample_split <- function(data, ..., p=0.8, n=1) {
  if (p < 0 | p > 1) {
    stop("p should be in [0,1]")
  }
  if (p < 0.5) {
    warning("You choose to keep less than half of the data for training and use the rest for validation; this is surprising. Consider using p>0.5.")
  }

  # convert input data to data.frame for modelr::resample
  data_df <- as.data.frame(data)

  split_in_parts <- function(n, p) {
    n1 <- round(n*p)
    n2 <- n - n1
    sample(rep.int(1:2, times=c(n1, n2)))
  }

  # compute the splits
  splits <- purrr::map_dfr(1:n, function(i) {
    split_ids <- data %>%
      dplyr::group_by(...) %>%
      dplyr::transmute(.split=split_in_parts(dplyr::n(), p)) %>%
      dplyr::pull(".split")

    dplyr::tibble(
      train = list(modelr::resample(data=data_df, idx=which(split_ids==1))),
      val   = list(modelr::resample(data=data_df, idx=which(split_ids==2))),
      repet = i
    )
  })

  # add a special class for further processing
  class(splits) <- c("resamples", class(splits))

  return(splits)
}
