#' Fit an xgboost model to *one* `resamples` object
#'
#' This is the "workhorse" function called by the different methods of xgb_fit.
#'
#' @param object of class resamples, created by a `resample_***()` function.
#' @param resp name of the response variable.
#' @param expl names of the explanatory variables.
#' @param params named list of parameters passed to `xgboost::xgb.train()`.
#' @param nrounds number of boosting rounds (i.e. number of trees).
#' @param verbose 0 = silent, 1 = display performance, 2 = more verbose.
#' @param weight a vector of observation-level weights, one per line of the training set.
#' @param nthread number of threads (cores) used to fir each model. This is set
#' to one by default to avoid conflict with parallelisation per resample (in
#' [`xgb_fit()`] which is more efficient). Set this to more than 1 when fitting
#' only one model.
#' @param ... other parameters passed to `xgboost::xgb.Train()``
#'
#' @returns The input object (a tibble of class `resamples`) with an
#' additional column called `model` containing the fitted model object.
#'
#' @importFrom dplyr `%>%`
#' @export
#' @examples
#' rs <- resample_identity(mtcars, 1)
#' m <- fit_one_xgb(object=rs,
#'   resp="mpg", expl=c("cyl", "hp", "qsec"),
#'   # pass hyperparameters as a list
#'   params=list(eta=0.1, max_depth=4),
#'   nrounds=20
#' )
#' m$model[[1]]$params
#' m <- fit_one_xgb(object=rs,
#'   resp="mpg", expl=c("cyl", "hp", "qsec"),
#'   # pass hyperparamters inline, through ...
#'   eta=0.1, max_depth=4,
#'   nrounds=20
#' )
#' m$model[[1]]$params
fit_one_xgb <- function(object, resp, expl, params=list(), nrounds, verbose=0,
                        weight=NULL, nthread=1, ...) {
  # TODO Add checks for arguments

  # extract training set, in dMatrix form, for xgboost
  train  <- as.data.frame(object$train, check.names=FALSE)
  dTrain <- xgboost::xgb.DMatrix(
    data =as.matrix(train[expl]),
    label=train[[resp]],
    # use mono-core here, by default
    # we will parallelise at a higher level
    nthread=nthread
  )
  if (!is.null(weight)) {
    xgboost::setinfo(dTrain, "weight", weight[as.integer(object$train[[1]])])
  }
  # TODO allow resp and expl to be unquoted, like in select()

  # do the same for the validation set, taking into the account the case when
  # it is missing
  val <- as.data.frame(object$val, check.names=FALSE)
  if (nrow(val) == 0) {
    val_list <- list()
  } else {
    val_list <- list(
      val = xgboost::xgb.DMatrix(
        data =as.matrix(val[expl]),
        label=val[[resp]],
        nthread=nthread
      )
    )
  }

  # train the model on the training set, with the provided params
  m <- xgboost::xgb.train(data=dTrain,
    params=params, nrounds=nrounds,
    watchlist=val_list,
    verbose=verbose,
    nthread=nthread,
    ...
  )

  # add model to this `resamples` object
  object$model <- list(m)
  return(object)
}
