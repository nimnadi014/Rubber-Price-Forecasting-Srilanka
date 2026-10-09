\# Rubber Price Forecasting in Sri Lanka



\## Project Overview



This research project compares traditional time-series analysis and machine learning techniques to forecast Standard Crepe Rubber (SCR1X) prices in Sri Lanka.



The main objective is to evaluate the forecasting performance of ARIMA and XGBoost models and identify the more effective model for predicting future rubber prices.



\## Research Objectives



\* Analyze historical Standard Crepe Rubber prices in Sri Lanka.

\* Develop an ARIMA time-series forecasting model.

\* Develop an XGBoost machine learning forecasting model.

\* Compare model performance using RMSE, MAE, and MAPE.

\* Forecast future SCR1X rubber prices.



\## Dataset



The dataset contains historical Standard Crepe Rubber prices in Sri Lanka.



\*\*Target variable:\*\* `SCR1X\_Price`



The data was prepared and processed before model development.



\## Methodology



\### ARIMA Model



The ARIMA model was developed using time-series analysis techniques, including stationarity testing and differencing where required.



\### XGBoost Model



The XGBoost model used lagged rubber prices as input features to learn historical price patterns and generate forecasts.



\## Model Performance Comparison



The following results were obtained from the research:



| Evaluation Metric | ARIMA | XGBoost |

| ----------------- | ----: | ------: |

| RMSE              | 67.65 |   26.83 |

| MAE               | 43.97 |   20.22 |

| MAPE              | 6.35% |   3.00% |



XGBoost achieved lower forecasting errors than ARIMA across all three evaluation metrics.



Compared with ARIMA, XGBoost reduced RMSE by approximately 60.34%, MAE by 54.01%, and MAPE by 52.76%.



\## Future Price Forecasts



The XGBoost model generated the following forecasts for the next three forecast periods:



| Forecast Period | Predicted Price (LKR) |

| --------------- | --------------------: |

| Period 1        |                708.74 |

| Period 2        |                707.17 |

| Period 3        |                703.88 |



The forecasts suggest a relatively stable price trend with a slight decrease over the forecast period.



\## Summary of Findings



The research found that both ARIMA and XGBoost could generate forecasts for SCR1X rubber prices in Sri Lanka. However, XGBoost demonstrated better forecasting performance, achieving lower RMSE, MAE, and MAPE values than ARIMA.



Based on the evaluated results, XGBoost was selected as the better-performing model in this study. The findings may provide useful insights for rubber farmers, traders, policymakers, and other stakeholders in the Sri Lankan rubber industry.



\## Tools and Technologies



\* R

\* Python

\* RStudio

\* ARIMA

\* XGBoost

\* Time-Series Analysis

\* Data Visualization



\## Project Type



Academic Data Science Research Project





