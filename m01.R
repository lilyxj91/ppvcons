## =====================================================================
##  M01: unconstrained logistic-regression baseline
## =====================================================================

fit_M01 <- function(X, Y, rho) {
  X <- as.matrix(X)
  model <- stats::glm(Y ~ ., data = data.frame(Y = Y, X),
                      family = stats::binomial())
  beta <- unname(stats::coef(model)[-1])
  score <- drop(X %*% beta)

  c_grid <- sort(score)
  TPF <- TNR <- numeric(length(c_grid))
  for (k in seq_along(c_grid)) {
    TPF[k] <- sum(score[Y == 1] >  c_grid[k]) / sum(Y == 1)
    TNR[k] <- sum(score[Y == 0] <= c_grid[k]) / sum(Y == 0)
  }
  FPF <- 1 - TNR
  denominator <- TPF * rho + FPF * (1 - rho)
  PV <- TPF * rho / denominator
  PV[denominator == 0] <- 0  # numerical convention when no positives are predicted

  best <- which(TPF + TNR == max(TPF + TNR))
  k <- best[which.max(PV[best])]

  list(beta = beta, c = c_grid[k],
       TPF = TPF[k], FPF = FPF[k], PV = PV[k], model = model)
}
