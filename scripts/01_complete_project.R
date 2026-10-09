install.packages("forecast")
install.packages("tseries")
install.packages("ggplot2")
install.packages("xgboost")
install.packages("Metrics")
install.packages("gridExtra")
library(forecast)
library(tseries)
library(ggplot2)
library(xgboost)
library(Metrics)
library(gridExtra)

# Project folders
setwd("C:/Users/nimna/OneDrive/Desktop/research/Rubber_Price_Forecasting_Srilanka")

dir.create("plots", showWarnings = FALSE)
dir.create("outputs", showWarnings = FALSE)

# Clear graphics devices
while (!is.null(dev.list())) dev.off()


# 01.LOAD DATA
df <- read.csv("C:/Users/nimna/OneDrive/Desktop/research/Rubber_Price_Forecasting_Srilanka/data/scr1x_rubber_prices.csv")

# Check column names
names(df) <- c("Date", "SCR1X_Price", "Production_Kgs", "Crude_Oil_Price", "USD_LKR")
df <- df[, c("Date", "SCR1X_Price", "Crude_Oil_Price", "USD_LKR")]

# Convert date
df$Date <- as.Date(df$Date, format = "%m/%d/%Y")

# Check missing values
sum(is.na(df))
df <- na.omit(df)

str(df)
head(df)
sum(is.na(df))

# 02.Create Time Series
rubber_ts <- ts(df$SCR1X_Price,
                frequency = 52)

# Time Series Plot

plot.ts(
  rubber_ts,
  ylab = "SCR1X Price (LKR)",
  col = "blue",
  lwd = 2,
  main = "Weekly SCR1X Rubber Prices"
)

#save
png("plots/time_series_plot.png", width = 1000, height = 600)

plot.ts(
  rubber_ts,
  ylab = "SCR1X Price (LKR)",
  col = "blue",
  lwd = 2,
  main = "Weekly SCR1X Rubber Prices"
)

dev.off()

# 03.Train_Test_Split
test_size <- 24

train_df <- df[1:(nrow(df) - test_size), ]
test_df  <- df[(nrow(df) - test_size + 1):nrow(df), ]

nrow(train_df)
nrow(test_df)

# Save train and test data
write.csv(train_df, "outputs/Rubber_Train.csv", row.names = FALSE)
write.csv(test_df, "outputs/Rubber_Test_24weeks.csv", row.names = FALSE)

# Create time series objects
train_ts <- ts(train_df$SCR1X_Price, frequency = 52)
test_ts  <- ts(test_df$SCR1X_Price, frequency = 52)

# Plot train data
png(
  "plots/train_data_plot.png",
  width = 1000,
  height = 600
)

plot.ts(
  train_ts,
  main = "Weekly SCR1X Rubber Price in Sri Lanka - Train Data",
  ylab = "SCR1X Price (LKR)",
  xlab = "Time",
  col = "blue",
  lwd = 2
)

dev.off()

# Plot test data
png("plots/test_data_plot.png",
    width = 1000,
    height = 600)

plot.ts(
  test_ts,
  main = "Weekly SCR1X Rubber Price in Sri Lanka - Test Data Last 24 Weeks",
  ylab = "SCR1X Price (LKR)",
  xlab = "Time",
  col = "red",
  lwd = 2
)

dev.off()

# 04.Decomposition
decomp_train <- decompose(train_ts, type = "additive")

# Decomposition Plot 
plot(decomp_train)

png("plots/decomposition_train_plot.png",
    width = 1200,
    height = 800)

plot(decomp_train)

dev.off()

# 05.ADF Test
adf_result <- adf.test(train_ts)

print(adf_result)

# Save ADF Test Result
capture.output(
  adf_result,
  file = "outputs/adf_test_result.txt"
)

# 06.First Differencing
diff_train_ts <- diff(train_ts)

# Show differenced plot
png("plots/first_difference_plot.png",
    width = 1000,
    height = 600)

plot.ts(
  diff_train_ts,
  main = "First Differenced Series",
  col = "blue",
  lwd = 2
)
dev.off()

# 07.ADF Test After First Differencing
adf_diff <- adf.test(diff_train_ts)

print(adf_diff)

# Save result to outputs folder
capture.output(
  adf_diff,
  file = "outputs/adf_diff_test_result.txt"
)

# 08.ACF & PACF on Differenced Series
png("plots/acf_diff_plot.png", width = 1000, height = 600)

acf(
  diff_train_ts,
  lag.max = 52,
  xaxt = "n",
  main = "ACF of First Differenced Series"
)

axis(1, at = seq(0, 52/52, by = 4/52),
     labels = seq(0, 52, by = 4))

mtext("Lag (Weeks)", side = 1, line = 3)

dev.off()

png("plots/pacf_diff_plot.png", width = 1000, height = 600)

pacf(
  diff_train_ts,
  lag.max = 52,
  xaxt = "n",
  main = "PACF of First Differenced Series"
)

axis(1, at = seq(0, 52/52, by = 4/52),
     labels = seq(0, 52, by = 4))

mtext("Lag (Weeks)", side = 1, line = 3)

dev.off()

# Fit ARIMA Model
best_model <- auto.arima(
  train_ts,
  d = 1,
  start.p = 1,
  start.q = 1,
  seasonal = FALSE,
  stepwise = FALSE,
  approximation = FALSE,
  ic = "aic",
  method = "ML"
)

summary(best_model)

# Save model summary
capture.output(
  summary(best_model),
  file = "outputs/arima_model_summary.txt"
)

# RESIDUAL DIAGNOSTICS
model_residuals <- residuals(best_model)

# 09.Ljung-Box test
ljung_box_test <- Box.test(
  model_residuals,
  lag = 20,
  type = "Ljung-Box"
)

print(ljung_box_test)

# Save Ljung-Box result
capture.output(
  ljung_box_test,
  file = "outputs/ljung_box_test.txt"
)

# 10.Residual time plot
png("plots/arima_residuals_plot.png", width = 1000, height = 600)

autoplot(model_residuals) +
  ggtitle("Residuals from ARIMA Model") +
  xlab("Time") +
  ylab("Residuals")

dev.off()

# 11.LOG TRANSFORM

train_log <- log(train_ts)

fit_log <- auto.arima(train_log)

summary(fit_log)

capture.output(
  summary(fit_log),
  file = "outputs/log_arima_model_summary.txt"
)

res_log <- residuals(fit_log)

# Show residual plot
plot(
  res_log,
  main = "Residuals of Log-ARIMA Model",
  ylab = "Residuals"
)
abline(h = 0, col = "red")

# Save residual plot
png("plots/Residual_Plot_LogARIMA.png", width = 800, height = 600)

plot(
  res_log,
  main = "Residuals of Log-ARIMA Model",
  ylab = "Residuals"
)
abline(h = 0, col = "red")

dev.off()

# Show residual ACF
Acf(
  res_log,
  main = "Residual ACF - Log-ARIMA"
)

# Save residual ACF
png("plots/Residual_ACF_LogARIMA.png", width = 800, height = 600)

Acf(
  res_log,
  main = "Residual ACF - Log-ARIMA"
)

dev.off()

# Show residual PACF
Pacf(
  res_log,
  main = "Residual PACF - Log-ARIMA"
)

# Save residual PACF
png("plots/Residual_PACF_LogARIMA.png", width = 800, height = 600)

Pacf(
  res_log,
  main = "Residual PACF - Log-ARIMA"
)

dev.off()

# Ljung-Box log
ljung_log <- Box.test(
  res_log,
  lag = 20,
  type = "Ljung-Box",
  fitdf = length(fit_log$coef)
)

print(ljung_log)

capture.output(
  ljung_log,
  file = "outputs/ljung_box_test_log.txt"
)


# 12.BOX-COX TRANSFORM + ARIMA

# 1.BOX-COX Lambda
lambda <- BoxCox.lambda(train_ts)

train_bc <- BoxCox(train_ts, lambda)

fit_bc <- auto.arima(train_bc)

summary(fit_bc)

capture.output(
  summary(fit_bc),
  file = "outputs/boxcox_arima_model_summary.txt"
)

capture.output(
  lambda,
  file = "outputs/boxcox_lambda.txt"
)

res_bc <- residuals(fit_bc)

# Show residual plot
plot(
  res_bc,
  main = "Residual Plot - BoxCox ARIMA",
  ylab = "Residuals"
)
abline(h = 0, col = "red")

# Save residual plot
png("plots/Residual_Plot_BoxCox.png", width = 800, height = 600)

plot(
  res_bc,
  main = "Residual Plot - BoxCox ARIMA",
  ylab = "Residuals"
)
abline(h = 0, col = "red")

dev.off()

# Show residual ACF
Acf(
  res_bc,
  main = "Residual ACF - BoxCox ARIMA"
)

# Save residual ACF
png("plots/Residual_ACF_BoxCox.png", width = 800, height = 600)

Acf(
  res_bc,
  main = "Residual ACF - BoxCox ARIMA"
)

dev.off()

# Show residual PACF
Pacf(
  res_bc,
  main = "Residual PACF - BoxCox ARIMA"
)

# Save residual PACF
png("plots/Residual_PACF_BoxCox.png", width = 800, height = 600)

Pacf(
  res_bc,
  main = "Residual PACF - BoxCox ARIMA"
)

dev.off()

# Ljung-Box Box-Cox
ljung_bc <- Box.test(
  res_bc,
  lag = 20,
  type = "Ljung-Box",
  fitdf = length(fit_bc$coef)
)

print(ljung_bc)

capture.output(
  ljung_bc,
  file = "outputs/ljung_box_test_boxcox.txt"
)

# 13.METRICS FUNCTION
calc_metrics <- function(actual, forecast) {
  rmse <- sqrt(mean((actual - forecast)^2, na.rm = TRUE))
  mae  <- mean(abs(actual - forecast), na.rm = TRUE)
  mape <- mean(abs((actual - forecast) / actual), na.rm = TRUE) * 100
  return(c(RMSE = rmse, MAE = mae, MAPE = mape))
}

# Calculate metrics
m_arima <- calc_metrics(test_df$SCR1X_Price, fc_arima$mean)
m_log   <- calc_metrics(test_df$SCR1X_Price, fc_log_original)
m_bc    <- calc_metrics(test_df$SCR1X_Price, fc_bc_original)

rmse_arima <- m_arima["RMSE"]
mae_arima  <- m_arima["MAE"]
mape_arima <- m_arima["MAPE"]

rmse_log <- m_log["RMSE"]
mae_log  <- m_log["MAE"]
mape_log <- m_log["MAPE"]

rmse_bc <- m_bc["RMSE"]
mae_bc  <- m_bc["MAE"]
mape_bc <- m_bc["MAPE"]


# 13.MODEL COMPARISON
metrics_df <- data.frame(
  Model = c("Original ARIMA", "Log-ARIMA", "Box-Cox ARIMA"),
  RMSE = c(rmse_arima, rmse_log, rmse_bc),
  MAE = c(mae_arima, mae_log, mae_bc),
  MAPE = c(mape_arima, mape_log, mape_bc),
  Ljung_Box_p_value = c(
    ljung_arima$p.value,
    ljung_log$p.value,
    ljung_bc$p.value
  )
)

print(metrics_df)

write.csv(
  metrics_df,
  "outputs/arima_model_comparison_metrics.csv",
  row.names = FALSE
)

# Save comparison output as text
capture.output(
  metrics_df,
  file = "outputs/arima_model_comparison_metrics.txt"
)