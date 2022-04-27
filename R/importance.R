#' Compute variable importance for each resample
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
#' # look at importance for the first resample
#' m$importance[[1]]
#' # summarise across resamples
#' summarise_importance(m)
#' # plot the summarised importance
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

#' Summarise variable importance across resamples
#'
#' @param object an object output by `importance()`, which contains an `importance` column.
#'
#' @returns A data.frame with variables and the mean and sd of their importance metrics.
#'
#' @export
#' @importFrom rlang .data
#' @family variable importance functions
#' @inherit importance examples
summarise_importance <- function(object) {
  dplyr::bind_rows(object$importance) %>%
    # compute average and sd of
    dplyr::group_by(.data$Feature) %>%
    dplyr::summarise(dplyr::across(.fns=c(mean=mean, sd=stats::sd))) %>%
    # TODO allow to use other functions like other summary approaches
    dplyr::ungroup() %>%
    # force display of variables in descending order
    dplyr::arrange(.data$Gain_mean) %>%
    dplyr::mutate(Feature=factor(.data$Feature, levels=.data$Feature))
}

#' Plot variable importance
#'
#' @inheritParams summarise_importance
#'
#' @returns A ggplot2 object.
#'
#' @export
#' @importFrom rlang .data
#' @import ggplot2
#' @family variable importance functions
#' @inherit importance examples
plot_importance <- function(object) {
  summarise_importance(object) %>%
    # plot
    ggplot() +
      geom_col(aes(y=.data$Feature, x=.data$Gain_mean)) +
      geom_errorbarh(aes(y=.data$Feature,
                         xmin=.data$Gain_mean-.data$Gain_sd,
                         xmax=.data$Gain_mean+.data$Gain_sd), height = 0.3) +
    labs(x="Mean+/-SD gain")
}
