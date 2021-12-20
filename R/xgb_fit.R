#' A generic function to fit an xgboost model
#'
#' @param object of class resamples.
#' @param ... passed to further methods.
#' @export
xgb_fit <- function(object, ...) {
  UseMethod("xgb_fit")
}

#' Fit an xgboost model for each row of a `resamples` object
#'
#' @param object of class resamples, created by a `resample_***()` function.
#' @param cores integer, number of cores to use for parallel computation.
#' @param ... passed to `fit_one_xgb()` and then later to `xgboost::xgb.Train()`
#'
#' @returns Like `fit_one_xgb()` but with several lines.
#'
#' @export
#' @examples
#' resample_boot(mtcars, 3) %>%
#'   xgb_fit(resp="mpg", expl=c("cyl", "hp", "qsec"),
#'     eta=0.1, max_depth=4, nrounds=20)
xgb_fit.resamples <- function(object, cores=1, ...) {
  # fit the model for each resample, in parallel
  res <- parallel::mclapply(1:nrow(object), function(i, ...) {
    fit_one_xgb(object[i,], ...)
  }, mc.cores=cores, ...)
  # recreate the full `resamples` object, with the added `model` column
  res <- do.call(dplyr::bind_rows, res)
  return(res)
}


#' Fit an xgboost model for each row of a `resamples.grid` object
#'
#' @param object of class resamples, created by the `param_grid()` function.
#' @param cores integer, number of cores to use for parallel computation.
#' @param ... passed to `fit_one_xgb()` and then later to `xgboost::xgb.Train()`
#'
#' @returns The input object with an additional column called
#' `model` containing the fitted model object.
#'
#' @returns Like fit_one_xgb() but with several lines and grouped per parameter
#' combination defined in the grid.
#'
#' @export
#' @examples
#' m <- resample_boot(mtcars, 2) %>%
#'   param_grid(eta=c(0.1, 0.5), max_depth=c(2, 4)) %>%
#'   xgb_fit(resp="mpg", expl=c("cyl", "hp", "qsec"),
#'           min_child_weight=2, nrounds=20)
#' m$model[[1]]$params
#' m$model[[3]]$params
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
