#' Compute classification quality metrics
#'
#' @param pred vector of predicted classes.
#' @param true vector of true classes.
#'
#' @returns A tibble with global accuracy, and averages (across classes) of
#'   precision, recall and F1-score.
#' @export
#' @examples
#' res <- mutate(mtcars, cyl=factor(cyl)) %>%
#'   resample_split(p=0.5) %>%
#'   xgb_fit(resp="cyl", expl=c("mpg", "hp", "qsec"), nrounds=20) %>%
#'   xgb_predict(fns=NULL)
#' classification_metrics(res$pred, res$cyl)
classification_metrics <- function(pred, true) {
  cr <- classification_report(pred, true)
  dplyr::tibble(
    accuracy = cr$precision[1],
    avg_recision  = cr$precision[2],
    avg_recall   = cr$recall[2],
    avg_f1   = cr$f1[2]
  )
}
