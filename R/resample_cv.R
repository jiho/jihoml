#' Split n items into k folds
#'
#' @param k integer, the number of folds.
#' @param n integer, the number of items.
#'
#' @return A vector of length n, containing a integer in 1:k = the fold identifier
#' @examples
#' joml:::split_in_folds(k=3, n=10)
#' table(joml:::split_in_folds(k=3, n=100))
split_in_folds <- function(n, k) {
  times <- ceiling(n/k)
  sample(rep.int(1:k, times), size=n)
}

#' Generate data resamples using cross validation
#'
#' @inheritParams resample_boot
#' @param k integer, the number of cross-validation folds.
#' @param n integer, the number of times to repeat the creation of k folds
#'          (n>1 means performing repeated cross validation).

#' @return A tibble with columns
#' - train : an object of class modelr::resample. It contains a pointer to .data
#'           and the indexes of the rows that are in the training set. To extract
#'           the training set, use `as.data.frame()`; to extract the row indexes
#'           use `as.integer()`
#' - val   : an object of class modelr::resample with the validation set = the
#'           fold that is not in the training set.
#' - fold  : integer, the fold index.
#' - rep   : integer, the repetition index.
#' @export
#' @examples
#' resample_cv(mtcars, k=3)
#' resample_cv(mtcars, k=3, rep=2)
#'
#' # stratified cross-val
#' rs  <- resample_cv(mtcars, k=4)
#' rss <- resample_cv(mtcars, k=4, gear)
#' sapply(rs$train, function(x) {sum(data.frame(x)$gear==4)})
#' # = variable number of occurrence of gear==4 in the training portion
#' sapply(rss$train, function(x) {sum(data.frame(x)$gear==4)})
#' # = reliable number of gear==4 in the training portion
resample_cv <- function(data, ..., k=3, n=1) {
  # checks
  k <- round(k)
  if (k <= 1) stop("The number of folds should be > 1.")
  if (k == 2) {
    warning("With only k=2 folds, you are splitting your data in half between training and validation. Using a smaller percentage of data for validation, with resample_split(), is probably more appropriate.")
  }
  if (n < 0) stop("The number of repetitions should be > 0.")

  # convert input data to data.frame for modelr::resample
  data_df <- as.data.frame(data)

  rfolds <- purrr::map_dfr(1:n, function(r, ...) {
    # define which observation goes in which fold,
    # in a stratified manner (by ...)
    fold_ids <- data %>%
      dplyr::group_by(...) %>%
      dplyr::transmute(.fold=split_in_folds(n=dplyr::n(), k=k)) %>%
      dplyr::pull(".fold")

    # split between train and val, for each fold
    folds <- purrr::map_dfr(1:k, function(i) {
      dplyr::tibble(
        train = list(modelr::resample(data=data_df, idx=which(fold_ids!=i))),
        val   = list(modelr::resample(data=data_df, idx=which(fold_ids==i))),
        fold = i
      )
    })

    # record the repetition
    folds$rep <- r

    return(folds)
  }, ...=...)

  # add a special class for further processing
  class(rfolds) <- c("resamples", class(rfolds))

  return(rfolds)
}
