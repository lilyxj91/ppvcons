## =====================================================================
##  M03: Kernel-smoothed method (Epanechnikov CDF)
## =====================================================================

fit_M03 <- function(X, Y, rho, tau, lambda, c_0, beta_init,
                    v = 1/3, epsilon = 1e-5,
                    maxit = 10000, reltol = 1e-8) {
  X <- as.matrix(X)
  X1 <- X[Y == 1, , drop = FALSE]
  X0 <- X[Y == 0, , drop = FALSE]
  n1 <- nrow(X1)
  n0 <- nrow(X0)
  h1 <- c_0 * n1^(-v)
  h0 <- c_0 * n0^(-v)
  c <- 1

  fit <- stats::optim(
    par = beta_init,
    fn = function(beta) {
      u1 <- pmax(-1, pmin((X1 %*% beta - c) / h1, 1))
      u0 <- pmax(-1, pmin((X0 %*% beta - c) / h0, 1))
      F1 <- 1/2 + 3/4*u1 - 1/4*u1^3
      F0 <- 1/2 + 3/4*u0 - 1/4*u0^3
      TPF_s <- sum(F1) / n1
      FPF_s <- sum(F0) / n0
      denominator <- TPF_s * rho + FPF_s * (1 - rho)
      PV_s <- if (denominator == 0) 0 else TPF_s * rho / denominator
      OB_s <- sum(log(F1 + epsilon)) / n1 +
              sum(log(1 - F0 + epsilon)) / n0
      L_s <- OB_s + lambda * (PV_s - tau)
      -L_s
    },
    method = "Nelder-Mead",
    control = list(maxit = maxit, reltol = reltol, trace = 0)
  )

  beta <- unname(fit$par)
  score1 <- drop(X1 %*% beta)
  score0 <- drop(X0 %*% beta)
  u1 <- pmax(-1, pmin((score1 - c) / h1, 1))
  u0 <- pmax(-1, pmin((score0 - c) / h0, 1))
  F1 <- 1/2 + 3/4*u1 - 1/4*u1^3
  F0 <- 1/2 + 3/4*u0 - 1/4*u0^3
  TPF_s <- sum(F1) / n1
  FPF_s <- sum(F0) / n0
  denominator <- TPF_s * rho + FPF_s * (1 - rho)
  PV_s <- if (denominator == 0) 0 else TPF_s * rho / denominator
  OB_s <- sum(log(F1 + epsilon)) / n1 +
          sum(log(1 - F0 + epsilon)) / n0

  TPF <- mean(score1 > c)
  FPF <- mean(score0 > c)
  denominator <- TPF * rho + FPF * (1 - rho)
  PV <- if (denominator == 0) 0 else TPF * rho / denominator

  list(beta = beta, c = c, h1 = h1, h0 = h0,
       TPF = TPF, FPF = FPF, PV = PV,
       TPF_s = TPF_s, FPF_s = FPF_s, PV_s = PV_s,
       OB_s = OB_s, L_s = -fit$value, convergence = fit$convergence)
}

## ---------------------------------------------------------------------
##  Choosing lambda and c_0 for M03 by cross-validation (the procedure of the paper)
## ---------------------------------------------------------------------

cv_M03 <- function(X, Y, rho, tau,
                   lambda_grid = c(0, 0.01, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 7.5, 10, 25, 50, 100, 200, 500),
                   c_0_grid = c(0.5, 1, 2), v = 1/3,
                   K = 5, repeats = 5, start_probs = seq(0.5, 0.95, by = 0.05), seed = 1, epsilon = 1e-5) {
  X <- as.matrix(X)
  c <- 1

  ## stratified folds, fixed before any fitting; each repeat is its own split
  fold <- matrix(0L, length(Y), repeats)
  for (r in seq_len(repeats)) {
    set.seed(seed + 100000 * (r - 1))
    for (y in 0:1) {
      id <- which(Y == y)
      id <- id[sample.int(length(id))]
      fold[id, r] <- rep(seq_len(K), length.out = length(id))
    }
  }

  ## logistic-regression starting values on a set of rows
  starts_on <- function(rows) {
    lr <- stats::glm(Y ~ ., data = data.frame(Y = Y[rows], X[rows, , drop = FALSE]), family = stats::binomial())
    beta_lr <- unname(stats::coef(lr)[-1])
    lapply(start_probs, function(p) beta_lr / unname(stats::quantile(drop(X[rows, , drop = FALSE] %*% beta_lr), p)))
  }
  ## hard metrics and the smoothed objective (bandwidths of the fit) on a set of rows
  metrics_on <- function(fit, rows) {
    s <- drop(X[rows, , drop = FALSE] %*% fit$beta); Yr <- Y[rows]
    TPF <- mean(s[Yr == 1] > c); FPF <- mean(s[Yr == 0] > c)
    denominator <- TPF * rho + FPF * (1 - rho)
    u <- pmax(-1, pmin((s - c) / ifelse(Yr == 1, fit$h1, fit$h0), 1))
    F <- 1/2 + 3/4 * u - 1/4 * u^3
    c(TPF = TPF, FPF = FPF, PV = if (denominator == 0) 0 else TPF * rho / denominator,
      OB = mean(log(F[Yr == 1] + epsilon)) + mean(log(1 - F[Yr == 0] + epsilon)))
  }

  ## every fold of every repeat from every start, for every (lambda, c_0)
  fold_starts <- lapply(seq_len(repeats), function(r) lapply(seq_len(K), function(q) starts_on(which(fold[, r] != q))))
  grid <- expand.grid(lambda = lambda_grid, c_0 = c_0_grid, start = seq_along(start_probs))
  grid$start_prob <- start_probs[grid$start]
  grid$complete <- FALSE; grid$min_PV <- NA; grid$mean_OB <- NA; grid$feasible <- FALSE
  for (g in seq_len(nrow(grid))) {
    PV <- OB <- matrix(NA_real_, repeats, K); ok <- TRUE
    for (r in seq_len(repeats)) for (q in seq_len(K)) {
      tr <- which(fold[, r] != q); va <- which(fold[, r] == q)
      f <- fit_M03(X[tr, , drop = FALSE], Y[tr], rho, tau, lambda = grid$lambda[g], c_0 = grid$c_0[g],
                   beta_init = fold_starts[[r]][[q]][[grid$start[g]]], v = v, epsilon = epsilon)
      if (f$convergence != 0 || any(!is.finite(f$beta))) { ok <- FALSE; break }
      m <- metrics_on(f, va); PV[r, q] <- m["PV"]; OB[r, q] <- m["OB"]
    }
    if (ok) {
      grid$complete[g] <- TRUE
      grid$min_PV[g] <- mean(apply(PV, 1, min))     # min over the K folds, averaged over repeats
      grid$mean_OB[g] <- mean(OB)                    # mean over all repeats x K folds
      grid$feasible[g] <- grid$min_PV[g] >= tau
    }
  }

  ## per start: CV winner(s), refitted on all rows from the same start
  all_starts <- starts_on(seq_along(Y))
  grid$cv_selected <- FALSE; grid$train_TPF <- NA; grid$train_FPF <- NA; grid$train_PV <- NA; grid$train_OB <- NA
  refits <- vector("list", nrow(grid))
  for (s in seq_along(start_probs)) {
    rows <- which(grid$start == s & grid$complete)
    if (!length(rows)) next
    feas <- rows[grid$feasible[rows]]
    pool <- if (length(feas)) feas else rows[grid$min_PV[rows] == max(grid$min_PV[rows])]
    winners <- pool[grid$mean_OB[pool] == max(grid$mean_OB[pool])]
    grid$cv_selected[winners] <- TRUE
    for (g in winners) {
      f <- fit_M03(X, Y, rho, tau, lambda = grid$lambda[g], c_0 = grid$c_0[g], beta_init = all_starts[[s]], v = v, epsilon = epsilon)
      if (f$convergence != 0 || any(!is.finite(f$beta))) next
      refits[[g]] <- f
      grid[g, c("train_TPF", "train_FPF", "train_PV", "train_OB")] <- metrics_on(f, seq_along(Y))
    }
  }

  ## across starts: training-set feasibility, then the training objective
  done <- which(!sapply(refits, is.null))
  if (!length(done)) stop("No candidate completed all folds and a full refit.")
  ok <- done[grid$train_PV[done] >= tau]
  if (length(ok)) {
    selection <- "feasible"
    pick <- ok[order(-grid$train_OB[ok], grid$lambda[ok], grid$c_0[ok], grid$start[ok])[1]]
  } else {
    selection <- "best_available_infeasible"
    pick <- done[order(-grid$train_PV[done], -grid$train_OB[done], grid$lambda[done], grid$c_0[done], grid$start[done])[1]]
  }
  grid$selected <- seq_len(nrow(grid)) == pick

  list(fit = refits[[pick]], lambda = grid$lambda[pick], c_0 = grid$c_0[pick], start_prob = grid$start_prob[pick],
       selection = selection, grid = grid, fold = if (repeats == 1) fold[, 1] else fold)
}
