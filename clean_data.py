import pathlib
import wfdb
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import glob
import data_loader


print("Starting")
name = "/home/stu25/project_2/new_data/data_labels_WFDB_Ningbo"
Y = pd.read_csv(name + ".csv")
dataset_All = data_loader.ECGDataset(Y, 500, 'sex', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
dataloader_All = DataLoader(dataset_All, batch_size=1, shuffle=False)
list_to_drop = []
print('start data filter')
for i, data in enumerate(dataloader_All):
    inputs, labels = data
    if torch.isnan(inputs).any():
        list_to_drop.append(i)
print('finish data filter')
print('found - {} bad data'.format(len(list_to_drop)))
new_name = name + "_clean.csv"
Y_new = Y.drop(list_to_drop)
Y_new.to_csv(new_name, index=False, encoding='utf-8-sig')