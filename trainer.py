# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:27:55 2023

@author: Shreyas Putty
"""
import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler

import utils


ROOT = Path(__file__)

main_path = ROOT.parents[0].__str__()

results_path = os.path.join(main_path, "Results")

basic = False
num_points = None

if basic:
    lookback = 3
    future_steps = 5
    batch_size = 1
    epochs = 50
    lr = 1e-2
else:
    num_points = 500
    lookback = 20
    future_steps = 25
    batch_size = 64
    epochs = 200
    lr = 1e-3

# define input sequence
timeseries_data = utils.generate_data(basic = basic, num_points = num_points,
                                      results_path = results_path)
print("timeseries_data -> ",  timeseries_data.shape)

scaler = MinMaxScaler(feature_range = (-1,1))
timeseries_data = scaler.fit_transform(timeseries_data)

timeseries_data = timeseries_data.squeeze()

training_size=int(len(timeseries_data)*0.7)

test_size = len(timeseries_data)-training_size

train_data, test_data = timeseries_data[0:training_size], timeseries_data[training_size:len(timeseries_data)]
print("train_data -> ", train_data.shape)
print("test_data -> ", test_data.shape)

utils.plot_ts_data(train_data, results_path, data_split = "Train")
utils.plot_ts_data(test_data, results_path, data_split = "Test")

# split into samples
X_train, y_train = utils.prepare_windowed_data(train_data, lookback)
print("X_train -> ", X_train.shape)
print("y_train -> ", y_train.shape)

X_test, y_test = utils.prepare_windowed_data(test_data, lookback)
print("X_test -> ", X_test.shape)
print("y_test -> ", y_test.shape)

train_dataset = utils.dataset_preparation(X_train, y_train)
test_dataset = utils.dataset_preparation(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

model = utils.VanillaLSTM(hidden_dim = 20,
                          input_dim = 1,
                          n_layers = 1,
                          output_dim = 1)

criterion = nn.MSELoss()
optimizer = optim.Adam(params = model.parameters(),
                       lr = lr)

best_loss = float('inf')

for epoch in range(epochs):
    train_loss = utils.train(train_loader, model, optimizer, criterion)
    test_loss, test_preds_all, test_labels_all = utils.test(test_loader,
                                                            model,
                                                            criterion)
    if epoch % 5 == 0:
        print(f"\nEpoch : {epoch}\tTr loss : {round(train_loss, 6)}\tTe loss : {round(test_loss, 6)}\n")
    
    if test_loss < best_loss:
        best_loss = test_loss
        
        utils.save_model(model, epoch, results_path)
        best_pred_dict = {"Actual" : test_labels_all,
                          "Predicted" : test_preds_all}
        

final_df = pd.DataFrame({**best_pred_dict})

Val_x_input = torch.FloatTensor((final_df["Actual"][-lookback:]).values)

loaded_model = utils.load_model(results_path)

future_pred = utils.predict_future(loaded_model, Val_x_input, lookback, future_steps)

timeseries_data, fut_fin_pred  = utils.final_results(timeseries_data,
                                                     future_pred,
                                                     final_df,
                                                     scaler,
                                                     future_steps,
                                                     results_path)

utils.final_excel(timeseries_data, results_path, name = "original_timeseries")
utils.final_excel(fut_fin_pred, results_path, name = "future_timeseries")