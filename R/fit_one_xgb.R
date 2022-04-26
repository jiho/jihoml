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
#' # regression
#' rs <- resample_identity(mtcars, 1)
#' m <- fit_one_xgb(object=rs,
#'   resp="mpg", expl=c("cyl", "hp", "qsec"),
#'   eta=0.1, max_depth=4,
#'   nrounds=20
#' )
#' m$model
#'
#' # classification
#' mtcarsf <- mutate(mtcars, cyl=factor(cyl))
#' rs <- resample_identity(mtcarsf, 1)
#' m <- fit_one_xgb(object=rs,
#'   resp="cyl", expl=c("mpg", "hp", "qsec"),
#'   eta=0.1, max_depth=4,
#'   nrounds=20
#' )
#' m$model
#'
#' # parameters can also be passed as a list
#' m_list <- fit_one_xgb(object=rs,
#'   resp="cyl", expl=c("mpg", "hp", "qsec"),
#'   params=list(eta=0.1, max_depth=4),
#'   nrounds=20
#' )
#' m$model[[1]]$params
#' m_list$model[[1]]$params
fit_one_xgb <- function(object, resp, expl, params=list(), nrounds, verbose=0,
                        weight=NULL, nthread=1, ...) {
  # TODO Add checks for arguments

  # extract training set, in dMatrix form, for xgboost
  train  <- as.data.frame(object$train, check.names=FALSE)
  if (is.factor(train[[resp]])) {
    # we are in classification mode, record additional things
    classif <- TRUE
    levels <- levels(train[[resp]])
    num_class <- length(levels)
    # convert to integer for xgboost
    train[[resp]] <- as.integer(train[[resp]]) - 1
  } else {
    classif <- FALSE
    num_class <- NULL
  }
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
    if (classif) {
      val[[resp]] <- as.integer(val[[resp]]) - 1
    }
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
    params=c(params, num_class=num_class), nrounds=nrounds,
    watchlist=val_list,
    verbose=verbose,
    nthread=nthread,
    ...
  )
  if (classif) {
    m$levels <- levels
  }

  # add model to this `resamples` object
  object$model <- list(m)
  return(object)
}
