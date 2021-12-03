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
#' # compute summary across resamples
#' resample_boot(mtcars, n=2) %>%
#'   xgb_fit(resp="mpg", expl=c("cyl", "hp", "qsec"),
#'           eta=0.1, max_depth=2, nrounds=10) %>%
#'   xgb_summarise_fit(fns=list(mean=mean, sd=sd))
#'
#' # when using a grid of parameters, the corresponding values of the parameters
#' # are preserved in the output.
#' x <- resample_boot(mtcars, n=2) %>%
#'   param_grid(eta=c(0.5, 1)) %>%
#'   xgb_fit(resp="mpg", expl=c("cyl", "hp", "qsec"),
#'           max_depth=2, nrounds=10) %>%
#'   xgb_summarise_fit()
#' print(x)
#' \dontrun{
#' ggplot(x) + geom_path(aes(x=iter, y=val_rmse_mean, colour=factor(eta)))
#' }
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
