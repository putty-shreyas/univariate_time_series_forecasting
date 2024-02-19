# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:02:31 2023

@author: Shreyas Putty
"""
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset

def generate_data(basic: bool, num_points: int = None,
                  results_path = None, trend: float = 0.02):
    
    # Generate a time series with random fluctuations
    if basic:
        time_series = [110, 212, 133, 146, 100, 172, 233, 196, 210,
                       233, 245, 110, 139, 122, 200, 211, 222, 230]
    else:
        # Randomly choose between -1 and 1 for direction
        direction = np.random.choice([-1, 1], size=num_points - 1)  
        
        time_series = [10]
    
        for i in range(1, num_points):
            value = time_series[i - 1] + direction[i - 1] * trend + np.random.normal(scale=5)
            time_series.append(round(value, 6))
    
    plot_ts_data(time_series, results_path, data_split = "Original Timeseries")
    
    return np.array(time_series).reshape(-1,1)

def plot_ts_data(time_series = None,
                 results_path: str = None,
                 data_split: str = None
                 ):
    # Plot the time series
    
    plt.grid()
    plt.plot(time_series, label = f"{data_split} data")
    plt.title(f'Randomly Varying Time Series - {data_split} data')
    plt.xlabel('Days')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig(os.path.join(results_path, f"{data_split.lower()}_data.png"), dpi = 400)
    plt.close()

# preparing independent and dependent features
def prepare_windowed_data(timeseries_data, lookback):
    X, y =[],[]
    for i in range(len(timeseries_data)):
        # find the end of this pattern
        end_ix = i + lookback
        
        # check if we are beyond the sequence
        if end_ix > len(timeseries_data)-1:
            break
        
        # gather input and output parts of the pattern
        seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]

        X.append(seq_x)
        y.append(seq_y)
        
    return np.array(X), np.array(y)

def dataset_preparation(X, y):
    
    # reshape from 2d to 3d
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape(-1,1)

    return TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))

class VanillaLSTM(nn.Module):
    def __init__(self,
                 hidden_dim,
                 input_dim,
                 n_layers,
                 output_dim):
        super(VanillaLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                            batch_first = True)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.lrelu = nn.LeakyReLU()
        
    def forward(self, x):
        h0 = torch.zeros(self.n_layers, 
                          x.size(0), 
                          self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.n_layers, 
                          x.size(0), 
                          self.hidden_dim).requires_grad_()
        
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        hn = self.lrelu(hn.view(-1, self.hidden_dim))
        
        output = self.fc2(self.lrelu(self.fc1(hn)))
        
        return output
    
def train(train_loader, model, optimizer, criterion):
    
    losses = []
    model.train()
    
    for i, (inputs, labels) in enumerate(train_loader):

        optimizer.zero_grad()
        pred = model(inputs)

        loss = criterion(pred, labels)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
    epoch_loss = np.mean(losses)
    
        
    return epoch_loss

def test(test_loader, model, criterion):
    
    losses = []
    preds_all = None
    labels_all = None
    model.eval()
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
        
            pred = model(inputs)

            loss = criterion(pred, labels)
            losses.append(loss.item())
            
            pred = pred.detach()
            labels = labels.detach()
            
            if i == 0:
                preds_all = pred
                labels_all = labels
            else:
                preds_all = torch.cat((preds_all, pred), dim = 0)
                labels_all = torch.cat((labels_all, labels), dim = 0)
            
    epoch_loss = np.mean(losses)
    
    preds_all = preds_all.squeeze()
    labels_all = torch.reshape(labels_all, (-1,))
    
    return epoch_loss, preds_all, labels_all

def save_model(model, epoch, results_path):

    ckpt = {"model": model,
            "epoch": epoch}
        
    torch.save(ckpt, os.path.join(results_path, "best_model.pt"))
        
def load_model(results_path):
    
    ckpt = torch.load(os.path.join(results_path, "best_model.pt"))

    loaded_model = ckpt["model"]
    print(f"Model saved at Best Epoch: {ckpt['epoch']}")
    
    return loaded_model

def predict_future(model, x_input, lookback, future_steps):

    x_input = x_input.view(1, lookback, 1)
    
    temp_input = x_input.view(-1).tolist()
    lst_output = []
    
    with torch.no_grad():
        for i in range(future_steps):
            if len(temp_input) > lookback:
                x_input = torch.tensor(temp_input[1:], dtype=torch.float32).view(1, lookback, 1)
    
                yhat = model(x_input)
    
                temp_input.append(yhat.item())
                temp_input = temp_input[1:]
                lst_output.append(yhat.item())
            else:
                x_input = torch.tensor(temp_input, dtype=torch.float32).view(1, lookback, 1)
                yhat = model(x_input)
    
                temp_input.append(yhat.item())
                lst_output.append(yhat.item())
    
    return lst_output

def final_transformation(scaler, tensor):
    ## Transformback to original form
    rev_scaled_pred = scaler.inverse_transform(tensor.detach().numpy())
    
    return rev_scaled_pred

def plot_losses(results_path: str, loss: list, train: bool = True):
       
    task = ['Training' if train else 'Testing'][0]
    
    plt.plot(loss)
    plt.title(f"{task}_Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(os.path.join(results_path, f"{task.lower()}_losses.png"),
                dpi = 400)
    plt.close()

def final_results(timeseries_data, future_pred, final_df,
                  scaler, future_steps, results_path):
    
    test_labels = final_transformation(scaler, torch.FloatTensor(final_df["Actual"]).reshape(-1,1))
    
    test_predictions = final_transformation(scaler, torch.FloatTensor(final_df["Predicted"]).reshape(-1,1))

    fut_fin_pred = final_transformation(scaler, torch.FloatTensor(future_pred).reshape(-1,1))
    
    timeseries_data = final_transformation(scaler, torch.FloatTensor(timeseries_data).reshape(-1,1))

    current_day = np.arange(1,len(timeseries_data)+1)
    future_days = np.arange(len(timeseries_data), len(timeseries_data) + future_steps)

    plt.xlabel("Days")
    plt.ylabel("Values")
    plt.grid()
    plt.plot(current_day,timeseries_data.squeeze(), color = "g", label = "original data")
    plt.plot(future_days,fut_fin_pred.squeeze(), color = "r", label = "future prediction")
    plt.title("Quantity Forecast Prediction")
    plt.legend(loc = "best")
    plt.savefig(os.path.join(results_path, "quantity_forecast_prediction.png"),
                dpi = 400)
    plt.close()

    plt.xlabel("Days")
    plt.ylabel("Values")
    plt.grid()
    plt.plot(test_labels, color = 'g', label = " Actual Labels")
    plt.plot(test_predictions, color = 'r', label = "Predictions")
    plt.title("Model Performance")
    plt.legend(loc = "best")
    plt.savefig(os.path.join(results_path, "model_performance.png"), dpi = 400)
    plt.close()
    
    return timeseries_data, fut_fin_pred

def final_excel(timeseries_data, result_path, name:str):
    
    timeseries_data = timeseries_data.squeeze()
    df = pd.DataFrame({"Quantity": timeseries_data})
    df.index.name = "Time"
    
    df.to_excel(os.path.join(result_path, f"{name}.xlsx"))