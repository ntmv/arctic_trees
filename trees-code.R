# Libraries
library(tgp)
library(randomForest)
library(MASS)
library(e1071)
library(tidymodels)
library(ggplot2)
library(forcats)
# Reading in Wonderland data
input <- read.csv("ice_x.csv")
output <- read.csv("ice_y.csv")
# Keeping only ice velocity
x <- input
y <- output$IceVelocity
# Combining into one dataset
dat <- data.frame(cbind(x, y))
# Create cross-validation folds
set.seed(112)
folds  <- vfold_cv(dat, v = 10)
# For Bayesian CART model
cv_fit1 <- function(splits,...) { ZZ <- assessment(splits)$y
fit1 <- bcart(X = analysis(splits)[,-14],
              Z = analysis(splits)$y, XX = assessment(splits)[,-14])
rmse <- sqrt(mean((fit1$ZZ.mean - ZZ)^2))
return(rmse)
}
# Fitting entire training set
fit_cart <- bcart(X = x, Z = y)
plot(fit_cart)
# For Bayesian stationary model
cv_fit2 <- function(splits,...) { ZZ <- assessment(splits)$y
fit2 <- bgp(X = analysis(splits)[,-14],
            Z = analysis(splits)$y, XX = assessment(splits)[,-14],
            bprior = "b0", R = 2, corr = "exp")
rmse <- sqrt(mean((fit2$ZZ.mean - ZZ)^2))
return(rmse) }
# Fitting stationary GP
fit_bgp <-  bgp(X = x, Z = y,  bprior = "b0", corr = "exp", R = 2, trace = TRUE)
9
# For Bayesian treed GP model
cv_fit3 <- function(splits,...) { ZZ <- assessment(splits)$y
fit3 <- btgp(X = analysis(splits)[,-14], Z = analysis(splits)$y, XX = assessment(splits)[,-14],
             bprior = "b0", R = 2)
rmse <- sqrt(mean((fit3$ZZ.mean - ZZ)^2))
return(rmse) }
# For Bayesian treed GP with limiting linear model
cv_fit4 <- function(splits,...) { ZZ <- assessment(splits)$y
fit4 <- btgpllm(X = analysis(splits)[,-14], Z = analysis(splits)$y,
                XX = assessment(splits)[,-14], bprior = "b0", R = 2)
rmse <- sqrt(mean((fit4$ZZ.mean - ZZ)^2))
return(rmse)
}
# Running GP LLM model
# d[0/] means LLM is active
fit_llm <- btgpllm(X = x, Z = y, bprior = "b0", R = 2, trace = TRUE)
# MAP tree
png(filename = "map-tree.png", width = 16, height = 14, units = "cm", res = 300)
tgp.trees(fit_llm)
dev.off()
# A few plots
png(filename = "statgp.png", width = 20, height = 14, units = "cm", res = 300)
plot(fit_bgp, proj = c(3, 12))
dev.off()
png(filename = "treedgp.png", width = 20, height = 14, units = "cm", res = 300)
plot(fit_llm, proj = c(3, 12))
dev.off()
# Extracting RMSE’s
res_cv_train <-
  folds %>%
  mutate(res_rmse1 = map(splits, .f = cv_fit1) ,
         res_rmse2 = map(splits, .f = cv_fit2),
         res_rmse3 = map(splits, .f = cv_fit3),
         res_rmse4 = map(splits, .f = cv_fit4))
# Mean of RMSE’s
rmse_cart <- sqrt(mean((unlist(res_cv_train$res_rmse1))^2))
rmse_gp <- sqrt(mean((unlist(res_cv_train$res_rmse2))^2))
rmse_tgp <- sqrt(mean((unlist(res_cv_train$res_rmse3))^2))
rmse_tgpllm <- sqrt(mean((unlist(res_cv_train$res_rmse4))^2))
var_cart <- sqrt(var((unlist(res_cv_train$res_rmse1))))/sqrt(10)
var_gp <- sqrt(var((unlist(res_cv_train$res_rmse2))))/sqrt(10)
var_tgp <- sqrt(var((unlist(res_cv_train$res_rmse3))))/sqrt(10)
var_tgpllm <- sqrt(var((unlist(res_cv_train$res_rmse4))))/sqrt(10)
# CV varying mtry and ntree
ice.tune <-  tune.randomForest(y ~., data = dat, mtry = 1:13, ntree=100*1:8,
                               tunecontrol = tune.control(sampling = "cross",cross=5))

# Random forests
# Setting up model with mtry = 13 and ntree = 300 rf_mod <-
rand_forest(mtry = 13, ntree = 300) %>%
  set_engine("ranger") %>%
  set_mode("regression")
# Running random forests
rf_fit <-
  rf_mod %>%
  fit(y ~ ., data = dat)
# Workflow for cross-validation
rf_wf <-
  workflow() %>%
  add_model(rf_mod) %>%
  add_formula(y ~ .)
# Fit on 10-folds
rf_fit_rs <-
  rf_wf %>%
  fit_resamples(folds)
# Use tune package to collect metrics
collect_metrics(rf_fit_rs)
# Plotting RMSE’s
rmses <- c(rmse_cart, rmse_gp, rmse_tgp, collect_metrics(rf_fit_rs)$mean[1], rmse_tgpllm) sds <- c(var_cart, var_gp, var_tgp, collect_metrics(rf_fit_rs)$std_err[1], var_tgpllm) mod <- c("CART", "Stationary GP", "Treed GP", "Random Forest", "Treed LLM GP")
# Data framing
dat_plot <- data.frame(cbind(rmses, sds, mod))
# Convert class
dat_plot[, 1:2] <- lapply(dat_plot[, 1:2], as.numeric)
png(filename = "RMSE.png", width = 16, height = 14, units = "cm", res = 300) ggplot(dat_plot, aes(x= mod, y= rmses)) +
  geom_dotplot(binaxis=’y’, stackdir=’center’) +
  geom_crossbar(aes(ymin=rmses-sds, ymax=rmses+sds, col = mod, fill = mod, alpha = 0.01), width=.2,
                position=position_dodge(.9), alpha = 0.5) + theme_bw() +
  theme(legend.position="none") + aes(x = fct_inorder(mod)) + labs(col = "", x = "", y = "RMSE")
dev.off() 
