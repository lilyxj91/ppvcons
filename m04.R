## =====================================================================
##  M04 with logistic starting points: neural network, Epanechnikov smoothing
## =====================================================================

fit_M04_nn_start <- function(X, Y, rho, tau, lambda, c_0, v = 1/3, epochs = 10000L,
                             hidden_units = 0L, activation = "linear", dropout = 0,
                             use_bias = FALSE,
                             learning_rate = 0.01, batch_size = 256L, optimizer = "nadam",
                             patience = 1000L, min_delta = 0, epsilon = 1e-5, seed = NULL,
                             start_probs = c(0.55, 0.65, 0.85, 0.9, 0.95), beta_init = NULL) {
  X <- as.matrix(X)
  if (!is.numeric(X) || any(!is.finite(X)) || length(Y) != nrow(X) ||
      anyNA(Y) || !all(Y %in% c(0, 1)) || !all(c(0, 1) %in% Y))
    stop("X must be finite and Y matching 0/1 labels containing both groups.")
  if (length(c_0) != 1L || !is.finite(c_0) || c_0 <= 0 || length(v) != 1L || !is.finite(v) || v < 0)
    stop("Require one positive c_0 and a finite v >= 0; bandwidths are c_0*n_y^(-v).")
  if (any(!is.finite(c(rho, tau, lambda, epsilon, min_delta))) ||
      rho <= 0 || rho >= 1 || tau < 0 || tau > 1 || lambda < 0 || epsilon <= 0 || min_delta < 0)
    stop("Require 0 < rho < 1, 0 <= tau <= 1, lambda/min_delta >= 0 and epsilon > 0.")
  if (any(!is.finite(c(epochs, patience, batch_size))) || any(c(epochs, patience, batch_size) < 1) ||
      any(c(epochs, patience, batch_size) != floor(c(epochs, patience, batch_size))))
    stop("epochs, patience and batch_size must be positive integers.")
  if (!activation %in% c("linear", "selu", "relu"))
    stop("The logistic start can only be embedded through a linear, selu or relu activation.")
  lambda_a <- if (activation == "selu") 1.0507009873554805 else 1
  tf <- tensorflow::tf
  c <- 1
  n1 <- sum(Y == 1)
  n0 <- sum(Y == 0)
  h1 <- c_0 * n1^(-v)
  h0 <- c_0 * n0^(-v)
  h_train <- ifelse(Y == 1, h1, h0)

  ## Starting coefficients: training-only logistic slope / score quantile, as in M03's CV rule.
  if (is.null(beta_init)) {
    if (!is.numeric(start_probs) || !length(start_probs) || any(!is.finite(start_probs)) ||
        any(start_probs <= 0 | start_probs >= 1))
      stop("start_probs must be finite quantile probabilities strictly between 0 and 1.")
    initial <- stats::glm(Y ~ ., data = data.frame(Y = Y, X), family = stats::binomial())
    beta_lr <- unname(stats::coef(initial)[-1])
    if (!initial$converged || any(!is.finite(beta_lr)))
      stop("The training-only logistic initialization failed.")
    cutoffs <- unname(stats::quantile(drop(X %*% beta_lr), start_probs))
    if (any(!is.finite(cutoffs)) || any(cutoffs == 0))
      stop("A logistic score quantile is zero/non-finite; initial coefficients are undefined.")
    initial_beta <- t(sapply(cutoffs, function(q) beta_lr / q))
    if (ncol(X) == 1L) initial_beta <- matrix(initial_beta, ncol = 1L)
    initial_type <- "training_logistic"
  } else {
    initial_beta <- if (is.null(dim(beta_init))) matrix(beta_init, nrow = 1L) else as.matrix(beta_init)
    if (!is.numeric(initial_beta) || ncol(initial_beta) != ncol(X) || any(!is.finite(initial_beta)))
      stop("beta_init must be a finite p-vector or a matrix with one p-vector per row.")
    start_probs <- rep(NA_real_, nrow(initial_beta))
    initial_type <- "supplied"
  }
  n_starts <- nrow(initial_beta)
  starts <- vector("list", n_starts)
  start_results <- data.frame(start_id = seq_len(n_starts), start_prob = start_probs,
    initial_loss = NA_real_, loss = NA_real_, OB_g = NA_real_, PV_g = NA_real_,
    TPF = NA_real_, FPF = NA_real_, PV = NA_real_, best_epoch = NA_integer_,
    epochs_run = NA_integer_, stop_reason = "", selected = FALSE)
  best_fit <- NULL
  selected_start <- NA_integer_

  for (s in seq_len(n_starts)) {
    tf$keras$backend$clear_session()
    if (!is.null(seed)) {
      set.seed(seed)
      tf$keras$utils$set_random_seed(as.integer(seed))
    }
    ## Loss: negative penalized objective with batch-specific group means.
    loss <- reticulate::py_func(function(Y, g) {
      Y <- tf$cast(Y, g$dtype)
      n1_batch <- tf$reduce_sum(Y)
      n0_batch <- tf$reduce_sum(1 - Y)
      tf$debugging$assert_positive(n1_batch, message = "A mini-batch has no group-1 observations.")
      tf$debugging$assert_positive(n0_batch, message = "A mini-batch has no group-0 observations.")
      h_batch <- Y * h1 + (1 - Y) * h0
      u <- tf$clip_by_value((g - c) / h_batch, -1, 1)
      F <- 1/2 + 3/4*u - 1/4*u^3
      TPF_g <- tf$reduce_sum(Y * F) / n1_batch
      FPF_g <- tf$reduce_sum((1 - Y) * F) / n0_batch
      PV_g <- tf$math$divide_no_nan(TPF_g * rho, TPF_g * rho + FPF_g * (1 - rho))
      OB_g <- tf$reduce_sum(Y * tf$math$log(F + epsilon)) / n1_batch +
              tf$reduce_sum((1 - Y) * tf$math$log(1 - F + epsilon)) / n0_batch
      -OB_g - lambda * (PV_g - tau)
    })

    ## Network: zero hidden layers (no bias, as in fit_M04) or one hidden layer.
    model <- tf$keras$Sequential()
    model$add(tf$keras$layers$InputLayer(input_shape = reticulate::tuple(as.integer(ncol(X)))))
    if (hidden_units == 0) {
      model$add(tf$keras$layers$Dense(units = 1L, activation = activation, use_bias = FALSE))
    } else {
      model$add(tf$keras$layers$Dense(units = as.integer(hidden_units), activation = activation, use_bias = use_bias))
      if (dropout > 0) model$add(tf$keras$layers$Dropout(rate = dropout))
      model$add(tf$keras$layers$Dense(units = 1L, activation = "linear", use_bias = use_bias))
    }
    opt <- switch(match.arg(optimizer, c("nadam", "adam")),
      nadam = tf$keras$optimizers$legacy$Nadam(learning_rate = learning_rate),
      adam  = tf$keras$optimizers$legacy$Adam(learning_rate = learning_rate))
    model$compile(optimizer = opt, loss = loss)

    ## Embed the linear start so that the initial score equals beta_start' x.
    beta_start <- initial_beta[s, ]
    w <- model$get_weights()
    if (hidden_units == 0) {
      w[[1]][, 1] <- beta_start / lambda_a
    } else if (!use_bias) {
      w[[1]][, 1] <- beta_start
      w[[2]][, 1] <- 0; w[[2]][1, 1] <- 1 / lambda_a
    } else {
      m_shift <- 1 - min(X %*% beta_start)
      w[[1]][, 1] <- beta_start; w[[2]][1] <- m_shift
      w[[3]][, 1] <- 0; w[[3]][1, 1] <- 1 / lambda_a; w[[4]][1] <- -m_shift
      if (hidden_units > 1) w[[2]][2:hidden_units] <- 0
    }
    model$set_weights(w)
    X_tensor <- tf$convert_to_tensor(X, dtype = tf$float32)
    g_init <- as.numeric(model$`__call__`(X_tensor, training = FALSE)$numpy())
    u0 <- pmax(-1, pmin((g_init - c) / h_train, 1)); F0 <- 1/2 + 3/4*u0 - 1/4*u0^3
    TPF0 <- mean(F0[Y == 1]); FPF0 <- mean(F0[Y == 0]); den0 <- TPF0 * rho + FPF0 * (1 - rho)
    initial_loss <- -(mean(log(F0[Y == 1] + epsilon)) + mean(log(1 - F0[Y == 0] + epsilon))) -
                    lambda * ((if (den0 == 0) 0 else TPF0 * rho / den0) - tau)
    if (!is.finite(initial_loss)) stop("Non-finite loss at the starting point.")

    ## Epoch 0 is a valid checkpoint; an update cannot discard a better initial fit.
    ## Complete training objective after each epoch, at fixed weights, dropout off.
    history <- data.frame(epoch = seq_len(epochs), batch_loss = NA_real_, loss = NA_real_, OB_g = NA_real_, PV_g = NA_real_)
    best_loss <- reference_loss <- initial_loss
    best_weights <- model$get_weights()
    best_epoch <- epochs_run <- wait <- 0L
    stop_reason <- "max_epochs"
    on_epoch_end <- reticulate::py_func(function(epoch, logs) {
      e <- as.integer(epoch) + 1L
      batch_loss <- as.numeric(logs[["loss"]])
      g <- as.numeric(model$`__call__`(X_tensor, training = FALSE)$numpy())
      if (any(!is.finite(c(batch_loss, g)))) stop("NN training produced NA/Inf; no fitted model is returned.")
      u <- pmax(-1, pmin((g - c) / h_train, 1))
      F <- 1/2 + 3/4*u - 1/4*u^3
      TPF_g <- mean(F[Y == 1]); FPF_g <- mean(F[Y == 0])
      denominator <- TPF_g * rho + FPF_g * (1 - rho)
      PV_g <- if (denominator == 0) 0 else TPF_g * rho / denominator
      OB_g <- mean(log(F[Y == 1] + epsilon)) + mean(log(1 - F[Y == 0] + epsilon))
      LL_g <- -OB_g - lambda * (PV_g - tau)
      if (!is.finite(LL_g)) stop("Non-finite whole-training loss.")
      history[e, c("batch_loss", "loss", "OB_g", "PV_g")] <<- c(batch_loss, LL_g, OB_g, PV_g)
      epochs_run <<- e
      if (LL_g < best_loss) { best_loss <<- LL_g; best_epoch <<- e; best_weights <<- model$get_weights() }
      if (LL_g < reference_loss - min_delta) { reference_loss <<- LL_g; wait <<- 0L } else wait <<- wait + 1L
      if (wait >= patience) { stop_reason <<- "training_loss_plateau"; model$stop_training <- TRUE }
      invisible(NULL)
    })
    model$fit(x = X, y = matrix(Y, ncol = 1), epochs = as.integer(epochs), batch_size = as.integer(batch_size),
              shuffle = TRUE, verbose = 0L,
              callbacks = list(tf$keras$callbacks$TerminateOnNaN(), tf$keras$callbacks$LambdaCallback(on_epoch_end = on_epoch_end)))
    if (is.null(best_weights)) stop("No epoch produced a finite training objective.")
    history <- rbind(data.frame(epoch = 0L, batch_loss = NA_real_, loss = initial_loss,
      OB_g = -initial_loss - lambda * ((if (den0 == 0) 0 else TPF0 * rho / den0) - tau),
      PV_g = if (den0 == 0) 0 else TPF0 * rho / den0),
      history[seq_len(epochs_run), , drop = FALSE])
    model$set_weights(best_weights)

    ## Metrics at the returned weights: smoothed (_g) and hard.
    g <- as.numeric(model$`__call__`(X_tensor, training = FALSE)$numpy())
    u <- pmax(-1, pmin((g - c) / h_train, 1))
    F <- 1/2 + 3/4*u - 1/4*u^3
    TPF_g <- mean(F[Y == 1]); FPF_g <- mean(F[Y == 0])
    denominator <- TPF_g * rho + FPF_g * (1 - rho)
    PV_g <- if (denominator == 0) 0 else TPF_g * rho / denominator
    OB_g <- mean(log(F[Y == 1] + epsilon)) + mean(log(1 - F[Y == 0] + epsilon))
    L_g <- OB_g + lambda * (PV_g - tau)
    TPF <- mean(g[Y == 1] > c); FPF <- mean(g[Y == 0] > c)
    denominator <- TPF * rho + FPF * (1 - rho)
    PV <- if (denominator == 0) 0 else TPF * rho / denominator

    fit_s <- list(model = model, beta = model$get_weights(), c = c, c_0 = c_0, v = v, h1 = h1, h0 = h0,
                  n1_train = n1, n0_train = n0, use_bias = use_bias, hidden_units = hidden_units,
                  activation = activation, beta_start = beta_start, g_init = g_init, initial_loss = initial_loss,
                  TPF = TPF, FPF = FPF, PV = PV, PV_defined = denominator > 0,
                  TPF_g = TPF_g, FPF_g = FPF_g, PV_g = PV_g, OB_g = OB_g, L_g = L_g, LL_g = -L_g,
                  epochs_run = epochs_run, best_epoch = best_epoch, stop_reason = stop_reason, history = history)
    starts[[s]] <- fit_s
    starts[[s]]$model <- NULL
    start_results[s, c("initial_loss", "loss", "OB_g", "PV_g", "TPF", "FPF", "PV", "best_epoch", "epochs_run")] <-
      c(initial_loss, fit_s$LL_g, fit_s$OB_g, fit_s$PV_g, fit_s$TPF, fit_s$FPF, fit_s$PV, fit_s$best_epoch, fit_s$epochs_run)
    start_results$stop_reason[s] <- fit_s$stop_reason
    if (is.null(best_fit) || fit_s$LL_g < best_fit$LL_g) { best_fit <- fit_s; selected_start <- s }
  }
  start_results$selected[selected_start] <- TRUE
  best_fit$selected_start <- selected_start
  best_fit$initialization <- initial_type
  best_fit$initial_beta <- initial_beta
  best_fit$start_results <- start_results
  best_fit$starts <- starts
  best_fit
}

## ---------------------------------------------------------------------
##  Choosing lambda, c_0 and the network structure for M04 by cross-validation
## ---------------------------------------------------------------------

cv_M04 <- function(X, Y, rho, tau,
                   lambda_grid = c(0, 0.01, 0.1, 0.5, 1, 5, 10, 25, 50, 100, 200, 500),
                   c_0_grid = c(0.5, 1), v = 1/3,
                   structures = data.frame(hidden_units = c(0L, 1L, 2L), activation = c("linear", "selu", "selu")),
                   K = 5, repeats = 5, start_probs = c(0.55, 0.65, 0.85, 0.9, 0.95), seed = 1, epsilon = 1e-5,
                   fit_fun = fit_M04_nn_start, ...) {
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

  ## hard metrics and the smoothed objective (bandwidths of the fit) on a set of rows
  metrics_on <- function(fit, rows) {
    s <- as.numeric(fit$model$predict(X[rows, , drop = FALSE], verbose = 0L)); Yr <- Y[rows]
    if (any(!is.finite(s))) stop("Predictions contain NA/Inf.")
    TPF <- mean(s[Yr == 1] > c); FPF <- mean(s[Yr == 0] > c)
    denominator <- TPF * rho + FPF * (1 - rho)
    u <- pmax(-1, pmin((s - c) / ifelse(Yr == 1, fit$h1, fit$h0), 1))
    F <- 1/2 + 3/4 * u - 1/4 * u^3
    c(TPF = TPF, FPF = FPF, PV = if (denominator == 0) 0 else TPF * rho / denominator,
      OB = mean(log(F[Yr == 1] + epsilon)) + mean(log(1 - F[Yr == 0] + epsilon)))
  }
  one_fit <- function(rows, g, s, fit_seed) {
    tryCatch(fit_fun(X[rows, , drop = FALSE], Y[rows], rho, tau, lambda = grid$lambda[g], c_0 = grid$c_0[g], v = v,
                     hidden_units = grid$hidden_units[g], activation = as.character(grid$activation[g]),
                     epsilon = epsilon, seed = fit_seed, start_probs = start_probs[s], ...),
             error = function(e) NULL)
  }

  ## every fold of every repeat from every start, for every (lambda, c_0, structure)
  hp <- expand.grid(lambda = lambda_grid, c_0 = c_0_grid)
  grid <- do.call(rbind, lapply(seq_len(nrow(structures)), function(a) data.frame(hp, structures[a, ], row.names = NULL)))
  grid$candidate <- seq_len(nrow(grid))
  grid <- do.call(rbind, lapply(seq_along(start_probs), function(s) data.frame(grid, start = s, start_prob = start_probs[s])))
  rownames(grid) <- NULL
  grid$complete <- FALSE; grid$min_PV <- NA; grid$mean_OB <- NA; grid$feasible <- FALSE
  for (g in seq_len(nrow(grid))) {
    PV <- OB <- matrix(NA_real_, repeats, K); ok <- TRUE
    for (r in seq_len(repeats)) for (q in seq_len(K)) {
      f <- one_fit(which(fold[, r] != q), g, grid$start[g], seed + (r - 1) * (K + 1) + q)
      if (is.null(f)) { ok <- FALSE; break }
      m <- metrics_on(f, which(fold[, r] == q)); PV[r, q] <- m["PV"]; OB[r, q] <- m["OB"]
    }
    if (ok) {
      grid$complete[g] <- TRUE
      grid$min_PV[g] <- mean(apply(PV, 1, min))     # min over the K folds, averaged over repeats
      grid$mean_OB[g] <- mean(OB)                    # mean over all repeats x K folds
      grid$feasible[g] <- grid$min_PV[g] >= tau
    }
  }

  ## per start: CV winner(s), refitted on all rows from the same start
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
      f <- one_fit(seq_along(Y), g, s, seed + repeats * (K + 1))
      if (is.null(f)) next
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
    pick <- ok[order(-grid$train_OB[ok], grid$lambda[ok], grid$c_0[ok], grid$candidate[ok], grid$start[ok])[1]]
  } else {
    selection <- "best_available_infeasible"
    pick <- done[order(-grid$train_PV[done], -grid$train_OB[done], grid$lambda[done], grid$c_0[done], grid$candidate[done], grid$start[done])[1]]
  }
  grid$selected <- seq_len(nrow(grid)) == pick

  list(fit = refits[[pick]], lambda = grid$lambda[pick], c_0 = grid$c_0[pick],
       hidden_units = grid$hidden_units[pick], activation = as.character(grid$activation[pick]), start_prob = grid$start_prob[pick],
       selection = selection, grid = grid, fold = if (repeats == 1) fold[, 1] else fold)
}
