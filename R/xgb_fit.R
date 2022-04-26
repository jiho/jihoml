#' Fit an xgboost model for each row of a `resamples` object
#'
#' @param object of class `resamples` (created by a `resample_***()` function) or
#'   of class `resamples_grid` (created by `param_grid()`).
#' @param ... passed to `fit_one_xgb()` and then later to `xgboost::xgb.Train()`
#'
#' @inherit fit_one_xgb return
#'
#' @export
#' @examples
#' # regression
#' resample_boot(mtcars, 3) %>%
#'   xgb_fit(resp="mpg", expl=c("cyl", "hp", "qsec"),
#'     eta=0.1, max_depth=4, nrounds=20)
#'
#' resample_boot(mtcars, 2) %>%
#'   param_grid(eta=c(0.1, 0.2)) %>%
#'   xgb_fit(resp="mpg", expl=c("cyl", "hp", "qsec"),
#'     max_depth=4, nrounds=20)
#'
#' # classification
#' mtcarsf <- mutate(mtcars, cyl=factor(cyl))
#' resample_boot(mtcarsf, 3) %>%
#'   xgb_fit(resp="cyl", expl=c("mpg", "hp", "qsec"),
#'     eta=0.1, max_depth=4, nrounds=20)
xgb_fit <- function(object, ...) {
  UseMethod("xgb_fit")
}

#' @rdname xgb_fit
#'
#' @param cores integer, number of cores to use for parallel computation.
#'
#' @method xgb_fit resamples
#' @export
xgb_fit.resamples <- function(object, cores=1, ...) {
  # fit the model for each resample, in parallel
  res <- parallel::mclapply(1:nrow(object), function(i, ...) {
    fit_one_xgb(object[i,], ...)
  }, mc.cores=cores, ...)
  # recreate the full `resamples` object, with the added `model` column
  res <- do.call(dplyr::bind_rows, res)
  return(res)
}


#' @rdname xgb_fit
#'
#' @method xgb_fit resamples_grid
#' @export
xgb_fit.resamples_grid <- function(object, cores=1, ...) {
  # extract the names of parameters
  all_names <- names(object)
  train_idx <- which(all_names == "train")
  param_names <- all_names[1:(train_idx-1)]

  # fit the model for each resample, in parallel
  res <- parallel::mclapply(1:nrow(object), function(i, ...) {
    fit_one_xgb(object[i,], params=as.list(object[i,param_names]), ...)
  }, ..., mc.cores=cores)

  # recreate the full `resamples` object, with the added `model` column
  res <- do.call(dplyr::bind_rows, res) %>%
    # and group it by parameter combination
    dplyr::group_by(dplyr::across(dplyr::all_of(param_names)))
  return(res)
}
