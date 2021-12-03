#' Generate data resamples using bootstrap
#'
#' @param .data data.frame, the data to resample.
#' @param n integer, number of bootstraps.
#' @param ... unquoted names of columns of .data to stratify by. Usually they are discrete variables.
#'
#' @return A tibble with columns
#' - train : an object of class modelr::resample. The boostrapped data.
#' - val   : an object of class modelr::resample. The out-of-bag data (i.e. the
#'           rows of .data not selected in the current bootstrap).
#' - boot  : integer, the bootstrap number.
#'
#' @export
#' @examples
#' resample_boot(mtcars, n=2)
#' rs <- resample_boot(mtcars, n=2)
#' data.frame(rs$train[1])
#' # = some rows are repeated
#' data.frame(rs$val[1])
#' # = all these rows are not above
#'
#' # stratified bootstrap
#' rs  <- resample_boot(mtcars, n=10)
#' rss <- resample_boot(mtcars, n=10, gear)
#' sapply(rs$train, function(x) {sum(data.frame(x)$gear==4)})
#' # = variable number of occurrence of gear==4 in the training set
#' sapply(rss$train, function(x) {sum(data.frame(x)$gear==4)})
#' # = reliable number of gear==4 in the training set
resample_boot <- function(.data, n=10, ...) {
  if (n < 0) stop("The number of bootstraps should be > 0.")

  # convert input data to data.frame for modelr::resample
  data_df <- as.data.frame(.data)

  # add a row index (avoiding name clashes as much as possible)
  .data$..my_row_index.. <- 1:nrow(.data)

  # compute the bootstraps
  boots <- purrr::map_dfr(1:n, function(i) {
    in_bag <- .data %>%
      # stratify the boostrap
      dplyr::group_by(...) %>%
      # pick some rows in each stratum
      dplyr::group_map(.f=function(x, ...) {
        sample(x$..my_row_index.., size=nrow(x), replace=TRUE)
      }) %>%
      unlist()

    # use indices not selected in the bootstrap above for out-of-bag validation
    out_of_bag <- setdiff(.data$..my_row_index.., in_bag)

    dplyr::tibble(
      train = list(modelr::resample(data=data_df, idx=in_bag)),
      val   = list(modelr::resample(data=data_df, idx=out_of_bag)),
      boot = i
    )
  })

  # add a special class for further processing
  class(boots) <- c("resamples", class(boots))

  return(boots)
}
