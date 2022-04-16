#' Compute regression quality metrics
#'
#' @param pred numeric vector of predictions
#' @param true numeric vector of actual values
#'
#' @returns a tibble with RMSE, MAE, R2, correlation R2, correlation R2 on log(n+1) transformed data
#' @export
#' @examples
#' res <- resample_split(mtcars, p=0.8) %>%
#'   xgb_fit(resp="mpg", expl=c("cyl", "hp", "qsec"), nrounds=100) %>%
#'   xgb_predict(fns=NULL)
#' regression_metrics(res$pred, res$mpg)
regression_metrics <- function(pred, true) {
  dplyr::tibble(
    RMSE = MLmetrics::RMSE(pred, true),
    MAE  = MLmetrics::MAE(pred, true),
    R2   = MLmetrics::R2_Score(pred, true) * 100,
    R2_correl     = stats::cor(pred, true)^2 * 100,
    R2_correl_log = stats::cor(log1p(pred), log1p(true))^2 * 100
  )
}
pred_metrics <- regression_metrics
