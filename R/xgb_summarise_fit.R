#' Summarise the fit of xgboost models over resamples
#'
#' For each model in the `resamples` object, extract the evaluation log, created
#' during training. It gives the error metric as a function of the number of
#' boosting rounds. Compute summary statistics over the various resamples.
#'
#' @param object an object output by `xgb_fit()`, which contains a `model` column.
#' @param fns a named list of summary functions, to compute for each
#'            error metric in the evaluation log. If NULL or an empty list,
#'            just return the full log.
#'
#' @returns A tibble with columns
#' - the grouping columns in `object`. Ungroup the object before `xgb_summarise_fit()`
#'   if this is not the desired behaviour.
#' - `iter`    : the iteration, i.e. boosting round
#' - `***_+++` : where *** is the metric (e.g. val_rmse for the RMSE on the
#'               validation set) and +++ is the summary function (e.g. mean)
#' If the input was grouped, the grouping variables are preserved. Ungroup the
#' object beforehand if this is not the desired behaviour.
#'
#' @export
#' @importFrom rlang .data
#' @examples
#' # compute summary of fit metrics across 5 resamples
#' fit <- resample_boot(mtcars, n=5) %>%
#'   xgb_fit(resp="mpg", expl=c("cyl", "hp", "qsec"),
#'           eta=0.2, max_depth=2, nrounds=10)
#' fit_history <- xgb_summarise_fit(fit, fns=list(mean=mean, sd=sd))
#' plot(val_rmse_mean ~ iter, data=fit_history)
#' # this shows fitting is not finished, we should increase nrounds.
#'
#' # when using a grid of parameters, the corresponding values of the parameters
#' # are preserved in the output.
#' fit_history <- resample_boot(mtcars, n=5) %>%
#'   param_grid(eta=c(0.5, 1)) %>%
#'   xgb_fit(resp="mpg", expl=c("cyl", "hp", "qsec"),
#'           max_depth=2, nrounds=10) %>%
#'   xgb_summarise_fit()
#' print(fit_history)
#' \dontrun{
#' # then the fit history can be plotted separately for each, to choose the best
#' ggplot(fit_history) +
#'   geom_ribbon(aes(
#'     x=iter,
#'     ymin=val_rmse_mean-val_rmse_sd,
#'     ymax=val_rmse_mean+val_rmse_sd,
#'     fill=factor(eta)
#'   ), alpha=0.5) +
#'   geom_path(aes(x=iter, y=val_rmse_mean, colour=factor(eta)))
#' }
#'
#' # the same is true for classification of course
#' mtcarsf <- mutate(mtcars, cyl=factor(cyl))
#' fit_history <- resample_boot(mtcarsf, n=5) %>%
#'   xgb_fit(resp="cyl", expl=c("mpg", "hp", "qsec"),
#'           objective="multi:softprob",
#'           eta=0.6, max_depth=2, nrounds=10) %>%
#'   xgb_summarise_fit(fns=list(mean=mean, sd=sd))
#' plot(val_mlogloss_mean ~ iter, data=fit_history, type="b")
#' # -> We're over-fitting
xgb_summarise_fit <- function(object, fns=list(mean=mean, sd=stats::sd, se=se)) {
  if (is.null(object$model)) {
    stop("This input does not contain a `model` column. Have you forgotten to fit the model with xgb_fit()?")
  }

  # extract all logs
  log <- object %>%
    # NB: using do() here instead of map_dfr() directly allows to keep the
    #     possible groups in object. There are groups when fitting on a
    #     `resamples_grid` object.
    dplyr::do({
      purrr::map_dfr(.data$model, function(x) {x$evaluation_log})
    })

  if( nrow(log) == 0 ) {
    stop("No validation set was available, no fit statistics can be computed.")

  } else {
    # if summary functions are provided, use them
    if (length(fns) != 0) {
      log <- log %>%
        # NB: preserve existing groups again
        dplyr::group_by(.data$iter, .add=TRUE) %>%
        dplyr::summarise(dplyr::across(.fns=fns))
    }
  }
  return(dplyr::as_tibble(log))
}
