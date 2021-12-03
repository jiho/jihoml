#' Compute prediction quality metrics
#'
#' @param y_pred numeric vector of predictions
#' @param y_true numeric vector of actual values
#'
#' @returns a tibble with RMSE, MAE, R2, correlation R2, correlation R2 on log(n+1) transformed data
#' @export
#' @examples
#' res <- resample_split(mtcars, p=0.8) %>%
#'   xgb_fit(resp="mpg", expl=c("cyl", "hp", "qsec"), nrounds=100) %>%
#'   xgb_predict(fns=c())
#' pred_metrics(res$pred, res$mpg)
pred_metrics <- function(y_pred, y_true) {
  dplyr::tibble(
    RMSE = MLmetrics::RMSE(y_pred, y_true),
    MAE  = MLmetrics::MAE(y_pred, y_true),
    R2   = MLmetrics::R2_Score(y_pred, y_true) * 100,
    R2_correl     = stats::cor(y_pred, y_true)^2 * 100,
    R2_correl_log = stats::cor(log1p(y_pred), log1p(y_true))^2 * 100
  )
}
