#' Build a classification report
#'
#' Create a table giving precision, recall and F1-scores.
#'
#' @param pred vector of predicted classes.
#' @param true vector of true classes.
#' @param exclude vector of classes to exclude for average metrics computation.
#'   They are then marked by a `*`. This is typically used to exclude dominant
#'   classes that skew the average too much.
#'
#' @returns A data.frame with global and per class metrics.
#' @export
#' @examples
#' # fit and predict a classifier
#' res <- mutate(mtcars, cyl=factor(cyl)) %>%
#'   resample_cv(k=3) %>%
#'   xgb_fit(resp="cyl", expl=c("mpg", "hp", "qsec"), nrounds=30) %>%
#'   xgb_predict(fns=NULL)
#' res
#' classification_report(res$pred, res$cyl)
#' classification_report(res$pred, res$cyl, exclude=8)
classification_report <- function(pred, true, exclude=NULL) {
  cm <- confusion_matrix(pred, true) %>% as.matrix()

  # basic stats
  n <- sum(cm) # number of instances
  nc <- nrow(cm) # number of classes
  diag <- diag(cm) # number of correctly classified instances per class
  rowsums <- apply(cm, 1, sum) # number of instances per class
  colsums <- apply(cm, 2, sum) # number of predictions per class
  # p <- rowsums / n # distribution of instances over the actual classes
  # q <- colsums / n # distribution of instances over the predicted classes

  # metrics
  accuracy <- sum(diag) / n * 100

  precision <- diag / colsums * 100
  recall <- diag / rowsums * 100
  f1 <- 2 * precision * recall / (precision + recall)

  # classification report
  cr <- data.frame(
    n=table(true),
    precision,
    recall,
    f1
  )
  # reformat table
  names(cr)[1:2] <- c("class", "n")
  row.names(cr) <- NULL
  cr$class <- as.character(cr$class)

  # add global stats
  global <- bind_rows(
    data.frame(class="accuracy", n=NA, precision=accuracy, recall=accuracy, f1=accuracy),
    data.frame(class="avg", t(apply(cr[,-(1:2)], 2, mean))),
    data.frame(class="wgtd avg", t(apply(cr[,-(1:2)], 2, weighted.mean, w=cr$n)))
  )

  if (length(exclude) > 0) {
    cre <- cr[!cr$class %in% exclude,]
    global <- bind_rows(
      global,
      data.frame(class="excl* avg", t(apply(cre[,-(1:2)], 2, mean))),
      data.frame(class="excl* wgtd avg", t(apply(cre[,-(1:2)], 2, weighted.mean, w=cre$n))),
    )
    cr[cr$class %in% exclude,"class"] <- paste0(cr[cr$class %in% exclude,"class"], "*")
  }

  cr <- bind_rows(global, cr)
  class(cr) <- c("cr", class(cr))
  return(cr)
}

#' @rdname classification_report
#' @export
cr <- classification_report

#' @rdname classification_report
#' @param object output of [`classification_report()`]
#' @export
print.cr <- function(object, digits=3) {
  out <- format(object, digits=digits)
  out <- lapply(out, function(x) {
    x[grepl("NA", x)] <- "-"
    return(x)
  }) %>% data.frame()
  n_head <- max(which(out$n == "-"))
  out <- rbind(
    out[1:n_head,],
    data.frame(class="***", n="***", precision="***", recall="***", f1="***"),
    out[(n_head+1):nrow(out),]
  )
  print(out, row.names=FALSE)
}

# #' @rdname classification_report
# #' @param object output of [`classification_report()`].
# #' @method show cr
# #' @export
# show.cr <- function(object) {
#   library("gt")
#   library("chroma")
#   object %>%
#     gt() %>%
#     data_color(columns=c(precision, recall, f1), colors=brewer_scale(name="RdYlGn")) %>%
#     fmt_percent(columns=c(precision, recall, f1), decimals=0, incl_space=TRUE) %>%
#     tab_row_group(label="global", rows=which(is.na(object$n)))
# }
# plot.cr <- show.cr
