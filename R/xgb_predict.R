#' Predict from an xgboost model at a given number of rounds, across resamples
#'
#' @param object an object output by `xgb_fit()`, which contains a `model` column.
#' @param newdata data.frame to predict, with the same variables as those used
#'                for fitting (and possibly others). When NULL, predict the
#'                validation data for each resample.
#' @param ntrees number of boosting trees to use in the prediction.
#'               Maps to the last bound of `iteration_range` in `xgboost::predict.xgb.Booster()`.
#' @param fns a named list of summary functions, to compute for the predictions
#'            of each observation, across resamples. If NULL or an empty list,
#'            just return the full predictions.
#' @param add_data boolean, whether to add the original data to the output
#'                 (defaults to TRUE which is practical to compute performance
#'                 metrics).
#' @param ... passed to xgboost::predict.xgb.Booster()
#'
#' @returns A tibble with columns
#' - the grouping columns in `object`. Ungroup the object before `xgb_predict()`
#'   if this is not the desired behaviour.
#' - the predictions as
#'    - `pred_***` where *** is a summary function (e.g. mean),
#'    or
#'    - `pred` when no summary function is chosen.
#' - the original data if `add_data` is TRUE.
#'
#' @export
#' @importFrom rlang .data
#' @examples
#' # fit models over 3 folds of cross-validation, with 6 rounds of boost each
#' fits <- resample_cv(mtcars, k=3) %>%
#'   xgb_fit(resp="mpg", expl=c("cyl", "hp", "qsec"),
#'           eta=0.1, max_depth=2, nrounds=30)
#' # compute the average predicted mpg over the 100 bootstraps, with 3 trees
#' res <- xgb_predict(fits, ntrees=20, fns=list(mean=mean))
#' res
#' # check that we have predicted all items in the dataset (should always be the
#' # case with cross validation)
#' nrow(res)
#' nrow(mtcars)
#' # compute Mean Squared Error at this tree number
#' sum(res$mpg-mean(res$mpg)^2)
#' sum((res$pred_mean - res$mpg)^2)/nrow(res)
#'
#' 1 - sum((res$mpg-res$pred_mean)^2)/sum((res$mpg-mean(res$mpg))^2)
#' cor(res$mpg, res$pred_mean)
#' MLmetrics::R2_Score(y_pred=res$pred_mean, y_true=res$mpg)
#'
#'
#' res <- resample_cv(mtcars, k=3) %>%
#'   param_grid(eta=c(0.1, 0.5)) %>%
#'   xgb_fit(resp="mpg", expl=c("cyl", "hp", "qsec"),
#'           max_depth=2, nrounds=30) %>%
#'   xgb_predict(ntrees=20, fns=list(mean=mean))
#' res %>% summarise(pred_metrics(pred_mean, mpg))
xgb_predict <- function(object, newdata=NULL, ntrees=NULL,
                        fns=list(mean=mean, sd=stats::sd, se=se),
                        add_data=TRUE, ...) {
  # checks
  if (is.null(object$model)) {
    stop("This input does not contain a `model` column. Have you forgotten to fit the model with xgb_fit()?")
  }
  # predict validation sets if the new data is NULL
  predict_val <- is.null(newdata)


  # predict
  preds <- object %>%
    # NB: use do() here to preserve grouping in the input object
    dplyr::do({
      # for each resample (in the current group)
      dplyr::rowwise(.data) %>% dplyr::do({
        if (predict_val) {
          # with no new data, predict the validation data
          newdata <- data.frame(.data$val)[.data$model$feature_names] %>% as.matrix()
          # and keep track of which rows of the original data that corresponds to
          rows <- as.integer(.data$val)
        } else {
          # predict the new data in its entirety (and therefore all its rows)
          newdata <- data.frame(newdata)[.data$model$feature_names] %>% as.matrix()
          rows <- 1:nrow(newdata)
        }

        # predict with the current model and the chosen number of boosting rounds
        pred <- stats::predict(
          .data$model,
          newdata=newdata,
          iteration_range=c(1,ntrees),
        )

        # return the prediction and the corresponding row indexes
        dplyr::tibble(..row=rows, pred)
      }) %>%
      dplyr::ungroup() %>%
      # make sure the data is in order
      dplyr::arrange(.data$..row)
    })

  # summarise the predictions for each data row, across resamples
  if (!is.null(fns)) {
    preds <- preds %>%
      dplyr::group_by(.data$..row, .add=TRUE) %>%
      dplyr::summarise(dplyr::across(.f=fns))
  }

  # combine predictions with the original data
  if (add_data) {
    # extract the data
    if (predict_val) {
      # we're predicting from the validation data, which is extracted from the
      # original data set, which was stored as:
      data <- object$train[1][[1]]$data
      # NB: this is the original, unshuffled data;
      #     it is the same for every resample (so we get it from the first one)
    } else {
      # we're predicting the new data
      data <- newdata
    }
    # add row indexes to the data
    data$..row <- 1:nrow(data)

    # join it to the prediction
    preds <- dplyr::left_join(preds, data, by="..row") %>%
      dplyr::select(-.data$..row)
  }

  return(preds)
}
