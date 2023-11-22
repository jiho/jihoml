#' Build a confusion matrix
#'
#' @param pred vector of predicted classes.
#' @param true vector of true classes.
#'
#' @returns A table object with true values in lines and predicted values in
#'   columns
#' @export
#' @examples
#' res <- mutate(mtcars, cyl=factor(cyl)) %>%
#'   resample_split(p=0.5) %>%
#'   xgb_fit(resp="cyl", expl=c("mpg", "hp", "qsec"), nrounds=20) %>%
#'   xgb_predict(fns=NULL)
#' confusion_matrix(res$pred, res$cyl)
confusion_matrix <- function(pred, true) {
  # make sure the factor levels are the same
  levels <- unique(c(pred, true))
  pred <- factor(pred, levels=levels)
  true <- factor(true, levels=levels)
  cm <- table(true=true, pred=pred)
  class(cm) <- c("cm", class(cm))
  return(cm)
}
#' @rdname confusion_matrix
# shortcut
cm <- confusion_matrix

#' Plot a confusion matrix
#'
#' @param x a confusion matrix built by [`confusion_matrix()`].
#' @param trans transformation function for the color scale.
#' @param ... passed to [`base::plot()`].
#'
#' @returns A ggplot object showing a heatmap with true values in lines and
#'   predicted values in columns.
#' @export
#' @importFrom rlang .data
#' @examples
#' res <- mutate(mtcars, cyl=factor(cyl)) %>%
#'   resample_split(p=0.5) %>%
#'   xgb_fit(resp="cyl", expl=c("mpg", "hp", "qsec"), nrounds=20) %>%
#'   xgb_predict(fns=NULL)
#' plot(confusion_matrix(res$pred, res$cyl))
plot.cm <- function(x, trans="log1p", ...) {
  as.data.frame(x) %>%
    ggplot() +
    geom_raster(aes(x=.data$pred, y=.data$true, alpha=.data$Freq), fill="darkblue") +
    scale_alpha_continuous(trans=trans, range=c(0,1), guide="none") +
    coord_fixed() +
    scale_x_discrete(expand=c(0,0), position="top") +
    scale_y_discrete(expand=c(0,0), limits=rev) +
    theme_light() +
    theme(
      axis.text.x=element_text(angle=65, hjust=0),
      panel.grid.major=element_line(colour="grey92")
    )
}

