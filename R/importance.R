#' Compute variable importance
#'
#' @param object an object output by `xgb_fit()`, which contains a `model` column.
#' @param cores integer, number of cores to use for parallel computation.
#' @param ... passed to `[xgboost::xgb.importance()]`.
#'
#' @returns A data.frame with variables and their importance metric.
#'
#' @export
#' @family variable importance functions
#' @examples
#' # fit a model on five bootstraps
#' m <- resample_boot(mtcars, 5) %>%
#'   xgb_fit(resp="mpg", expl=c("cyl", "hp", "qsec"),
#'     eta=0.1, max_depth=4, nrounds=20)
#' # compute variable importance for each model
#' m <- importance(m)
#' # look at importance for the first model
#' m$importance[[1]]
#' # plot the average importance
#' plot_importance(m)
importance <- function(object, cores=1, ...) {
  # extract importance metrics per model (in parallel)
  imp <- parallel::mclapply(1:nrow(object), function(i, ...) {
    xgboost::xgb.importance(model=object$model[[i]], ...) %>% as.data.frame()
  }, mc.cores=cores, ...)
  # store in the object
  object$importance <- imp
  return(object)
}

#' Plot variable importance
#'
#' @param object an object output by `importance()`, which contains an `importance` column.
#'
#' @returns A data.frame with variables and their importance metric.
#'
#' @export
#' @import ggplot2
#' @family variable importance functions
#' @examples
#' # fit a model on five bootstraps
#' m <- resample_boot(mtcars, 5) %>%
#'   xgb_fit(resp="mpg", expl=c("cyl", "hp", "qsec"),
#'     eta=0.1, max_depth=4, nrounds=20)
#' # compute variable importance for each model
#' m <- importance(m)
#' # plot the average importance
#' plot_importance(m)
plot_importance <- function(object) {
  dplyr::bind_rows(object$importance) %>%
    # compute average and sd of
    dplyr::group_by(Feature) %>%
    dplyr::summarise(dplyr::across(.fns=c(mean=mean, sd=stats::sd))) %>%
    ungroup() %>%
    # force display of variables in descending order
    dplyr::arrange(Gain_mean) %>%
    dplyr::mutate(Feature=factor(Feature, levels=Feature)) %>%
    # plot
    ggplot() +
      geom_col(aes(y=Feature, x=Gain_mean)) +
      geom_errorbarh(aes(y=Feature,
                         xmin=Gain_mean-Gain_sd,
                         xmax=Gain_mean+Gain_sd), height = 0.3) +
    labs(x="Mean+/-SD gain")
}
