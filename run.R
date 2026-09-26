## run.R — fit each method once on the demo data
##
##   m01.R  Logistic baseline      fit_M01(X, Y, rho)
##   m02.R  Exact                   fit_M02(X, Y, rho, tau, lambda, beta_init);            cv_M02() chooses lambda
##   m03.R  Kernel                  fit_M03(X, Y, rho, tau, lambda, c_0, beta_init);       cv_M03() chooses lambda and c_0
##   m04.R  Neural network (Keras)  fit_M04_nn_start(X, Y, rho, tau, lambda, c_0, hidden_units, activation, ...);  cv_M04() chooses lambda, c_0 and the structure

## ---- data --------------------------------------------------------------------------------
## demo_data.csv: cases and controls simulated from Scenario 1 of the paper; columns id, Y (1 = case), X1, X2, X3.
## For your own data: replace the file, keep a 0/1 column Y, and set x_cols, rho and tau.
demo_data <- read.csv("demo_data.csv")
set.seed(9876)
id_train <- c(sample(which(demo_data$Y == 1), 400), sample(which(demo_data$Y == 0), 400))
train <- demo_data[id_train, ]
test  <- demo_data[-id_train, ]
cat("training:", sum(train$Y == 1), "cases +", sum(train$Y == 0), "controls;  test:", sum(test$Y == 1), "cases +", sum(test$Y == 0), "controls\n")

x_cols <- c("X1", "X2", "X3")
X <- as.matrix(train[, x_cols]); Y <- train$Y
X_te <- as.matrix(test[, x_cols]); Y_te <- test$Y
rho <- 0.10  
tau <- 0.40   

## test-set sensitivity, specificity and PPV of the decision rule  score > c
evaluate <- function(score, c) {
  sensitivity <- mean(score[Y_te == 1] > c)
  specificity <- mean(score[Y_te == 0] <= c)
  ppv <- rho * sensitivity / (rho * sensitivity + (1 - rho) * (1 - specificity))
  c(sensitivity = sensitivity, specificity = specificity, ppv = ppv)
}
## ---- logistic baseline at matched operating points (the comparison used in the paper) -------
matched <- function(score, ref) {   # score: logistic test scores; ref: evaluate() of the constrained rule
  t_spec <- quantile(score[Y_te == 0], ref["specificity"])         # threshold with the same specificity
  t_sens <- quantile(score[Y_te == 1], 1 - ref["sensitivity"])     # threshold with the same sensitivity
  c(matched_sensitivity = mean(score[Y_te == 1] > t_spec), matched_specificity = mean(score[Y_te == 0] <= t_sens))
}



## ---- m01: logistic baseline ---------------------------------------------------------------
## Score beta'x from a logistic regression; the threshold c maximises Youden's index on the training data.
source("m01.R")
m01 <- fit_M01(X, Y, rho)
m01$beta                                                   
m01$c                                                    
c(train_sensitivity = m01$TPF, train_specificity = 1 - m01$FPF, train_ppv = m01$PV)
evaluate(drop(X_te %*% m01$beta), m01$c)



## ---- m02: Exact ---------------------------------------------------------------------------
source("m02.R")
lambda_grid <- c(0, 0.01, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 7.5, 10, 25, 50, 100, 200, 500)  
start_probs <- seq(0.5, 0.95, by = 0.05)  
K <- 5; repeats <- 5                     
cv02 <- cv_M02(X, Y, rho, tau, lambda_grid = lambda_grid, start_probs = start_probs, K = K, repeats = repeats, seed = 1)
c(lambda = cv02$lambda, start = cv02$start_prob); cv02$selection   
c(train_sensitivity = cv02$fit$TPF, train_specificity = 1 - cv02$fit$FPF, train_ppv = cv02$fit$PV)
ref02 <- evaluate(drop(X_te %*% cv02$fit$beta), cv02$fit$c)

rbind(Exact = ref02,
      Logistic = evaluate(drop(X_te %*% m01$beta), m01$c),
      `Logistic (matched to Exact)` = c(matched(drop(X_te %*% m01$beta), ref02), ppv = NA))



## ---- m03: Kernel --------------------------------------------------------------------------
source("m03.R")
c_0_grid <- c(0.5, 1, 2) 
cv03 <- cv_M03(X, Y, rho, tau, lambda_grid = lambda_grid, c_0_grid = c_0_grid, start_probs = start_probs, K = K, repeats = repeats, seed = 1)
c(lambda = cv03$lambda, c_0 = cv03$c_0, start = cv03$start_prob); cv03$selection
c(train_sensitivity = cv03$fit$TPF, train_specificity = 1 - cv03$fit$FPF, train_ppv = cv03$fit$PV)
ref03 <- evaluate(drop(X_te %*% cv03$fit$beta), cv03$fit$c)

rbind(Exact = ref02, Kernel = ref03,
      Logistic = evaluate(drop(X_te %*% m01$beta), m01$c),
      `Logistic (matched to Exact)`  = c(matched(drop(X_te %*% m01$beta), ref02), ppv = NA))



## ---- m04: neural network (TensorFlow / Keras) ---------------------------------------------
source("m04.R")
structures <- data.frame(hidden_units = c(0L, 1L, 2L),         
                         activation   = c("linear", "selu", "selu"))
lambda_grid_nn <- c(0, 1, 10)                                        
c_0_grid_nn    <- c(0.5, 1, 2)                                         
start_probs_nn <- c(0.7, 0.8, 0.9)                     
cv04 <- cv_M04(X, Y, rho, tau, lambda_grid = lambda_grid_nn, c_0_grid = c_0_grid_nn, structures = structures,
               start_probs = start_probs_nn, K = K, repeats = repeats, seed = 1,
               epochs = 1000L, patience = 100L)                      
c(lambda = cv04$lambda, c_0 = cv04$c_0, hidden_units = cv04$hidden_units, start = cv04$start_prob); cv04$activation; cv04$selection
c(train_sensitivity = cv04$fit$TPF, train_specificity = 1 - cv04$fit$FPF, train_ppv = cv04$fit$PV)
ref04 <- evaluate(as.numeric(cv04$fit$model$predict(X_te, verbose = 0L)), cv04$fit$c)

rbind(Exact = ref02, Kernel = ref03, NN = ref04,
      Logistic = evaluate(drop(X_te %*% m01$beta), m01$c),
      `Logistic (matched to Exact)`  = c(matched(drop(X_te %*% m01$beta), ref02), ppv = NA))


