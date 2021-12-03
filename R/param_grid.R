#' Define a parameter grid, to be explored through the fitting of resamples
#' 
#' @param object of class resamples, created by a `resample_***()` function.
#' @param ... named vectors of parameters, the combinations of which will
#'            constitute the grid
#'            
#' @returns An object of class `resamples_grid`, with one row per combination of
#' parameters value x resample (i.e. row of object). This can then be input to
#' `xgb_fit()` for training.
#'
#' @export
#' @examples 
#' rs <- resample_boot(mtcars, n=2)
#' param_grid(rs, eta=c(0.1, 0.5), max_depth=c(2, 4))
param_grid <- function(object, ...) {
  grid <- tidyr::crossing(..., object)
  class(grid) <- c("resamples_grid", "resamples", class(grid))
  return(grid)
}
