## Univariate Time Series Forecasting using LSTM network

Time Series Forecasting task performed on Univariate timeseries data using Vanilla LSTM architecture. Purpose of the project was to build a clean and efficient Python-based Deep Learning project with generated timeseries data and effective results storing. Model validation showcased through future forecasting of the data

## Features 
 - Option to generate random univariate timeseries data with varying trends or work with basic time series data to show efficient code operation
 - Future prediction of timeseries data highlighted and showcased

## Results and Summary 
<!-- Plot 1 --> 
<div align="center"> 
    <img src="https://raw.githubusercontent.com/putty-shreyas/univariate_time_series_forecasting/main/Results/original timeseries_data.png" alt="Plot 1" width="500" />
    <p>Original Timeseries Data</p>
</div>

<!-- Plot 2 -->
<div align="center">
    <img src="https://raw.githubusercontent.com/putty-shreyas/univariate_time_series_forecasting/main/Results/training_data.png" alt="Plot 2" width="500" />
    <p>Training Data Split</p>
</div>

<!-- Plot 3 -->
<div align="center">
    <img src="https://raw.githubusercontent.com/putty-shreyas/univariate_time_series_forecasting/main/Results/testing_data.png" alt="Plot 3" width="500" />
    <p>Testing Data Split</p>
</div>

<!-- Plot 4 -->
<div align="center">
    <img src="https://raw.githubusercontent.com/putty-shreyas/univariate_time_series_forecasting/main/Results/model_performance.png" alt="Plot 4" width="500" />
    <p>Model Performance on Testing Data</p>
</div>

<!-- Plot 5 -->
<div align="center">
    <img src="https://raw.githubusercontent.com/putty-shreyas/univariate_time_series_forecasting/main/Results/Training_losses.png" alt="Plot 4" width="500" />
    <p>Training Losses</p>
</div>

<!-- Plot 6 -->
<div align="center">
    <img src="https://raw.githubusercontent.com/putty-shreyas/univariate_time_series_forecasting/main/Results/Testing_losses.png" alt="Plot 4" width="500" />
    <p>Testing Losses</p>
</div>

<!-- Plot 7 -->
<div align="center">
    <img src="https://raw.githubusercontent.com/putty-shreyas/univariate_time_series_forecasting/main/Results/quantity_forecast_prediction.png" alt="Plot 4" width="500" />
    <p>Future Quantity Forecast Prediction</p>
</div>

Summary:
 - Vanilla LSTM model is good enough to perform a Univariate Time Series Forecasting task and captures the trend of the input data satisfactorily.
 - Both the losses stabilize in the end showcasing stable model training and performance.
 - Best Model is saved at the best performing epochs to be used for validation and future prediction later. 

## Getting Started
 - Run the trainer.py file in your environment and download the relevant packages if missing.
 - Possible packages that might need to be downloaded is openpyxl.
 - You would also need to create the folder "Results" to store the results.
```
run trainer.py
```

## About Me
I am Shreyas Putty, a M.Sc. Graduate in Data Science and Machine Learning and I am passionate about finding creative solutions through my knowledge and skills. I have 3+ years of experience in Python and am open to any new opportunities.

## Contact
We can connect through my email id - putty.shreyas@gmail.com and through my Linkedin - https://www.linkedin.com/in/shreyas-subhash-putty/