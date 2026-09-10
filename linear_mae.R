#maya powell
#dec 9 2025
#evaluating symcount and correcting counts

#load libraries
library(here)
library(ggplot2)
library(tidyverse)

eval_spat <- read.csv(here("Downloads","sym_counting_oki","eval_spatial_training_dec9.csv"))

#average error etc
summ <- eval_spat %>%
  summarise_if(is.numeric, mean, na.rm = TRUE)

pred_plot <- ggplot(eval_spat, aes(x = actual, y = pred)) +
  geom_point() +
  geom_smooth(method = "lm", col = "black")

mod <- lm(actual ~ pred, data = eval_spat) 
summary(mod)
#Multiple R-squared:  0.9893,	Adjusted R-squared:  0.989 
#Coefficients:
#(Intercept)         pred  
#      1.368        1.066  
#formula = 1.368 + 1.066x

#predict new data
eval_spat$pred_corrected <- predict(mod, newdata = eval_spat)

#error metrics
eval_spat$error_raw  <- eval_spat$pred - eval_spat$actual
eval_spat$error_corr <- eval_spat$pred_corrected - eval_spat$actual

mae_raw  <- mean(abs(eval_spat$error_raw),  na.rm = TRUE)
#5.357143
mae_corr <- mean(abs(eval_spat$error_corr), na.rm = TRUE)
#3.941768

#so yes the mean error does go down after correction

#plot corrected
pred_corr_plot <- ggplot(eval_spat, aes(x = actual, y = pred_corrected)) +
  geom_point() +
  geom_smooth(method = "lm", col = "black")
pred_corr_plot

mod_corr <- lm(actual ~ pred_corrected, data = eval_spat) 
summary(mod_corr)
#Multiple R-squared:  0.9893,	Adjusted R-squared:  0.989 

#plot both predicted and actual on one graph
p1 <- ggplot(eval_spat, aes(x = actual)) +
  # raw predictions
  geom_point(aes(y = pred, colour = "Raw predictions"), alpha = 0.6) +
  # corrected predictions
  geom_point(aes(y = pred_corrected, colour = "Corrected predictions"),
             alpha = 0.8, shape = 4) +  # shape 4 = x
  # 1:1 line
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "black") +
  scale_colour_manual(values = c("Raw predictions" = "black",
                                 "Corrected predictions" = "darkred")) +
  labs(
    x = "Actual count",
    y = "Predicted count",
    colour = NULL,
    title = "Prediction accuracy: raw vs corrected"
  ) +
  theme_bw() +
  theme(
    legend.position = "bottom"
  )
p1

#error plots comparison
err_plot <- ggplot(eval_spat, aes(x = actual, y = error_raw)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "black") +
  geom_point(alpha = 0.7) +
  labs(x = "Actual count", y = "Error (pred − actual)")
err_plot

err_plot_corr <- ggplot(eval_spat, aes(x = actual, y = error_corr)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "black") +
  geom_point(alpha = 0.7) +
  labs(x = "Actual count", y = "Error (corrected − actual)")
err_plot_corr
