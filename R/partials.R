#' Compute Univariate Partial Dependence
#'
#' Compute the partial dependence functions (i.e. marginal effects) for each
#' model in a resample.
#'
#' @param object an object output by `xgb_fit()`, which contains a `model`
#'   column.
#' @param expl a vector of explanatory variables to compute the partial
#'   dependence to.
#' @param cores integer, number of cores to use for parallel computation.
#' @param ... passed to [pdp::partial()]. Arguments of particular relevance are:
#'
#'   - `grid.resolution` : an integer giving the number of equally spaced points
#'   along continuous variables to compute the partial dependence at.
#'
#'   - `quantiles=TRUE` and `probs` (a vector of probabilities with values in
#'   \[0,1\]), to compute the partial dependence at those quantiles of the
#'   continuous explanatory variables.
#'
#' @details For each variable in `expl`, some target values are picked for
#'   continuous variables (along a grid or quantiles typically, see the
#'   arguments passed via `...`) and all levels are considered for categorical
#'   ones. For each target value of each target explanatory variable:
#'
#'   1. the training data is modified so that the target variable is made
#'   constant, equal to its target value, everywhere; all other explanatory
#'   variables remain unchanged.
#'
#'   2. the model predictions are computed for this new data set.
#'
#'   3. the predicted values are averaged, this gives `yhat` : the average
#'   prediction of the model for this value of the target variable.
#'
#' @returns The input object with a new column called `partial` containing a
#'   data.frame with the average prediction of the model (column `yhat`) for a
#'   given value (column `value`) of each explanatory variable (column
#'   `variable`).
#'
#' @export
#' @family partial dependence plots functions
#' @examples
#' # fit a model on 5 bootstraps
#' m <- resample_boot(mtcars, 5) %>%
#'   xgb_fit(resp="mpg", expl=c("cyl", "hp", "qsec"),
#'     eta=0.1, max_depth=4, nrounds=20)
#' # compute the 5 partial dependence and plot them
#' m <- partials(m, expl=c("hp", "cyl"))
#' plot_partials(m, fns=NULL)
#' # do the same with a finer grid
#' m <- partials(m, expl=c("hp", "cyl"), grid.resolution=50)
#' plot_partials(m, fns=NULL)
#' # or along quantiles
#' m <- partials(m, expl=c("hp", "cyl"), quantiles=TRUE, probs=0:20/20)
#' plot_partials(m, fns=NULL)
partials <- function(object, expl, cores=1, ...) {
  # check that explanatory variables exist
  all_expl <- object$model[[1]]$feature_names
  missing_expl <- setdiff(expl, all_expl)
  if (length(missing_expl) == 1) {
    stop("Variable ", missing_expl, " is not among the variables used to fit the model: ", paste0(all_expl, collapse=", "))
  } else if (length(missing_expl) > 1) {
    stop("Variables ", paste0(missing_expl, collapse=","), " are not among the variables used to fit the model: ", paste0(all_expl, collapse=", "))
  }

  # extract training data that is valid for all models
  # = union of all training sets
  all_train_indexes <- purrr::map(object$train, as.integer) %>% unlist() %>% unique()
  train_data <- object$train[[1]]$data[all_train_indexes,all_expl] %>% data.matrix()

  # compute the partial dependence plot for each mode
  pdps <- parallel::mclapply(1:nrow(object), function(i, ...) {
    # and each variable of interest
    purrr::map_dfr(expl, function(v, ...) {
      # compute partial dependence
      p <- pdp::partial(object$model[[i]], pred.var=v, train=train_data,
                        type="regression", plot=FALSE, ...)
      # reformat as data.frame
      p <- data.frame(p)
      names(p)[1] <- "value"
      p$variable <- v
      return(p)
    }, ...)
  }, mc.cores=cores, ...)

  # store PDPs as new column in object
  object$partial <- pdps

  return(object)
}

#' Plot (and summarise) partial dependence plots
#'
#' @param object an object output by `[partials()]`, which contains a `partial`
#'   column.
#' @param fns a list of summary functions; one should be called `location` and
#'   be used to compute the central location of the variable (e.g., mean,
#'   median, etc.); another should be called `spread` and be used to compute the
#'   spread around that location (e.g., sd, mad, etc.). When `fns` is `NULL`,
#'   the partial dependence line for each model is plotted.
#' @param rug boolean; whether to add a rug plot to show at which values of the
#'   explanatory variables the partial dependence is computed. This is most
#'   useful when partial dependence is computed at quantiles of the original
#'   data (`quantiles=TRUE` in `[partials()]`).
#'
#' @returns A ggplot2 object.
#'
#' @export
#' @family partial dependence plots functions
#' @import ggplot2
#' @examples
#' # fit a model on five bootstraps
#' m <- resample_boot(mtcars, 5) %>%
#'   xgb_fit(resp="mpg", expl=c("cyl", "hp", "qsec"),
#'     eta=0.1, max_depth=4, nrounds=20)
#' # plot all five partial dependence lines
#' partials(m, expl=c("hp", "cyl")) %>%
#'   plot_partials(fns=NULL, rug=FALSE)
#' # plot mean+/-sd among resamples, at given quantiles of the training data
#' partials(m, expl=c("hp", "cyl"), quantiles=TRUE) %>%
#'   plot_partials()
#' # do the same with median+/-mad
#' partials(m, expl=c("hp", "cyl"), quantiles=TRUE) %>%
#'   plot_partials(fns=list(location=median, spread=mad))
plot_partials <- function(object, fns=list(location=mean, spread=stats::sd), rug=TRUE) {
  df <- dplyr::bind_rows(object$partial, .id="id") %>%
    # force variable to be in the order they were specified when computing the pdp
    # NB: allows to start by the most important variable
    dplyr::mutate(variable=factor(variable, levels=unique(variable)))

  if (is.null(fns)) {
    # plot lines
    p <- ggplot(df)  +
      geom_path(aes(x=value, y=yhat, group=id),
                alpha=1/log(length(unique(df$id))+1))
    # NB: use a heuristic to find an appropriate transparency

  } else {
    # check consistency
    if ( any(! c("location", "spread") %in% names(fns)) ) {
      stop("'fns' needs to have one element named 'location' and one element named 'spread'.")
    }

    # compute summaries
    dfs <- df %>%
      group_by(variable, value) %>%
      summarise(loc=fns$location(yhat), spr=fns$spread(yhat), .groups="drop")

    # plot ribbon for spread and line for location
    p <- ggplot(dfs) +
      geom_ribbon(aes(x=value, ymin=loc-spr, ymax=loc+spr), alpha=0.4) +
      geom_path(aes(x=value, y=loc))
  }

  if (rug) {
    # extract the values at which the pdp was computed
    var_values <- unique(df[,c("variable", "value")])

    # add the rug to the plot
    n_unique <- max(table(var_values$variable))
    p <- p + geom_rug(aes(x=value), data=var_values, alpha=1/log10(n_unique+1))
    # NB: use a heuristic to find an appropriate transparency
  }

  p <- p +
    # facet per variable
    # and put variable name at the bottom
    facet_wrap(~variable, scales="free_x", strip.position="bottom") +
    # relabel y axis
    labs(y=expression(hat(y))) +
    # make the facet names look like axes titles
    theme(
      axis.title.x=element_blank(),
      strip.text.x=calc_element("axis.title.x", theme_get()),
      strip.switch.pad.wrap=unit(0, "npc"),
      strip.switch.pad.grid=unit(0, "npc"),
      strip.background = element_blank(),
      strip.placement = "outside"
    )

  return(p)
}
