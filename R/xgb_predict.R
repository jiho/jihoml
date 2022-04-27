#' Predict from an xgboost model at a given number of rounds, across resamples
#'
#' @param object an object output by `xgb_fit()`, which contains a `model`
#'   column.
#' @param newdata data.frame to predict, with the same variables as those used
#'   for fitting (and possibly others). When NULL, predict the validation data
#'   for each resample.
#' @param niter number of boosting iterations to use in the prediction. Maps to
#'   the last bound of `iterationrange` in `xgboost::predict.xgb.Booster()`.
#'   `niter`=0 or NULL means use all boosting rounds. Other values are
#'   equivalent to what is set in `nrounds` in [`xgb_fit()`].
#' @param fns a named list of summary functions, to compute over the predictions
#'   of each observation, across resamples (when there are more than one). If
#'   NULL, return all predictions. If "auto", the default, choose a function
#'   appropriate for the type of response variable: [`base::mean()`] for a
#'   numeric, continuous variable; [`majority_vote()`] for a factor.
#' @param add_data boolean, whether to add the original data to the output
#'   (defaults to TRUE which is practical to compute performance metrics).
#' @param ... passed to xgboost::predict.xgb.Booster()
#'
#' @returns A tibble with the grouping columns in `object` (ungroup the object
#'   before `xgb_predict()` if this is not the desired behaviour) and the
#'   prediction as
#'
#'   - `pred_***` where *** is a summary function (e.g. mean), or
#'
#'   - `pred` when no summary function is chosen. - the original data if
#'   `add_data` is TRUE.
#'
#' @export
#' @importFrom rlang .data
#' @examples
#' ## Regression
#'
#' # fit models over 4 folds of cross-validation, repeated 3 times
#' fits <- resample_cv(mtcars, k=4, n=3) %>%
#'   xgb_fit(resp="mpg", expl=c("cyl", "hp", "qsec"),
#'           eta=0.1, max_depth=2, nrounds=30)
#'
#' # compute the predicted mpg, with 20 trees only, and, by default average
#' # across the 3 repetitions
#' res <- xgb_predict(fits, niter=20)
#' head(res)
#'
#' # check that we have predicted all items in the dataset (should always be the
#' # case with cross validation)
#' nrow(res)
#' nrow(mtcars)
#' # compute the Root Mean Squared Error
#' sqrt( sum((res$pred_mean - res$mpg)^2) / nrow(res) )
#' # compute several regression metrics
#' regression_metrics(res$pred_mean, res$mpg)
#'
#' # examine the variability among the 3 repetitions of the cross validation
#' # do not average over the repetitions => we get 3x32 lines
#' res <- xgb_predict(fits, niter=20, fns=NULL)
#' nrow(res)
#' # compute the mean but also the standard deviation and error across repetitions
#' res <- xgb_predict(fits, niter=20, fns=list(mean=mean, sd=sd, se=se))
#' head(res)
#'
#'
#' ##  Classification
#'
#' # fit models over 4 folds of cross-validation, repeated 3 times
#' mtcarsf <- mutate(mtcars, cyl=factor(cyl))
#' fits <- resample_cv(mtcarsf, k=4, n=3) %>%
#'   xgb_fit(resp="cyl", expl=c("mpg", "hp", "qsec"),
#'           eta=0.1, max_depth=2, nrounds=30)
#'
#' # compute the predicted number of cylinders (cyl) but with only 15 of the 30
#' # rounds; by default, use the majority vote across the 3 repetitions
#' res <- xgb_predict(fits, niter=15)
#' head(res)
#' # compute accuracy
#' sum(res$pred_maj == res$cyl) / nrow(res)
#' # compute several global classification metrics
#' classification_metrics(res$pred_maj, res$cyl)
#'
#' # use a different objective for classification
#' fits <- resample_cv(mtcarsf, k=4, n=3) %>%
#'   xgb_fit(resp="cyl", expl=c("mpg", "hp", "qsec"),
#'           objective="multi:softprob",
#'           eta=0.1, max_depth=2, nrounds=30)
#' # because the objective is softprob, we predict the probability for each level
#' res <- xgb_predict(fits)
#' head(res)
#' # get the predicted class
#' res$max_prob_idx <- res %>%
#'   select(starts_with("pred_")) %>%
#'   apply(1, which.max)
#' res$pred_cyl <- refactor(res$max_prob_idx-1L, levels=levels(mtcarsf$cyl))
#' head(res)
#' # NB: refactor() uses 0-based indexing and needs integers, hence the -1L
xgb_predict <- function(object, newdata=NULL, niter=NULL,
                        fns="auto", add_data=TRUE, ...) {
  # checks
  if (is.null(object$model)) {
    stop("This input does not contain a `model` column. Have you forgotten to fit the model with xgb_fit()?")
  }
  # predict validation sets if the new data is NULL
  predict_val <- is.null(newdata)
  # check the nature of fns
  if (is.character(fns)) {
    if (fns != "auto") {
      stop("fns should be a list of functions or 'auto'")
    }
  } else {
    if (! all(sapply(fns, is.function))) {
      stop("fns should be a list of functions or 'auto'")
    }
  }

  # predict
  preds <- object %>%
    # NB: use do() here to preserve grouping in the input object
    dplyr::do({
      # for each resample (in the current group)
      dplyr::rowwise(.data) %>% dplyr::do({
        if (predict_val) {
          # with no new data, predict the validation data
          newdata <- data.frame(.data$val, check.names=FALSE)[.data$model$feature_names]
          # and keep track of which rows of the original data that corresponds to
          rows <- as.integer(.data$val)
        } else {
          # predict the new data in its entirety (and therefore all its rows)
          newdata <- data.frame(newdata, check.names=FALSE)[.data$model$feature_names]
          rows <- 1:nrow(newdata)
        }

        # predict with the current model and the chosen number of boosting rounds
        # if the number of boosting rounds is not set, then use all
        if (is.null(niter)) {niter <- 0}
        pred <- stats::predict(
          .data$model,
          newdata=data.matrix(newdata),
          iterationrange=c(1,niter+1),
          reshape=TRUE
        )

        if (!is.null(.data$model$levels) & is.null(dim(pred))) {
          # classification with softmax objective
          # convert the prediction back into a factor
          pred <- refactor(as.integer(pred), levels=.data$model$levels)
        } else if (!is.null(.data$model$levels) & !is.null(dim(pred))) {
          # classification with softprob objective
          pred <- as.data.frame(pred)
          names(pred) <- paste0("pred_", .data$model$levels)
        }

        # return the prediction and the corresponding row indexes
        dplyr::tibble(..row=rows, pred)
      }) %>%
      dplyr::ungroup() %>%
      # make sure the data is in order
      dplyr::arrange(.data$..row)
    })

  if (!is.null(fns) & nrow(object) > 1) {
    preds <- dplyr::group_by(preds, .data$..row, .add=TRUE)
    if (is.list(fns)) {
      # use the specified functions
      preds <- dplyr::summarise(preds, dplyr::across(.fns=fns))
    } else {
      # determine which function to use automatically
      preds <- preds %>%
        dplyr::summarise(
          dplyr::across(
            .cols=tidyselect::vars_select_helpers$where(is.numeric),
            .fns=list(mean=mean)
          ),
          dplyr::across(
            .cols=tidyselect::vars_select_helpers$where(is.factor),
            .fns=list(maj=majority_vote)
          )
        )
    }
  }

  # combine predictions with the original data
  if (add_data) {
    # extract the data
    if (predict_val) {
      # we're predicting the validation data, which is extracted from the
      # original data set, which was stored as:
      data <- object$val[1][[1]]$data
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
