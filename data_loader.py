import pandas as pd
import numpy as np
import wfdb
import ast
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset


class ECGDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
            self,
            df,
            sampling_rate,
            label_type,  # sex/age
            list_of_leads = None
    ):
        self.df = df
        self.sampling_rate = sampling_rate
        self.label_type = label_type
        self.test_fold = 10
        self.val_fol = 9
        self.list_of_leads = list_of_leads

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i: int):
        df = self.df.iloc[i]

        if self.sampling_rate == 100:
            data = wfdb.rdsamp('/home/stu25/project/data/'+ df.filename_lr)
        else:
            data = wfdb.rdsamp('/home/stu25/project/data/'+ df.filename_hr)
        signal, _ = data
        data = np.array(signal)
        if self.list_of_leads == None:
            data_pad = np.zeros((data.shape[0] + 120, data.shape[1]))
            data_pad[60:-60, :] = data
        else:
            data_pad = np.zeros((data.shape[0] + 120, len(self.list_of_leads)))
            data_pad[60:-60, :] = data[:,np.array(self.list_of_leads)]
        signal = torch.Tensor(data_pad)

        if self.label_type == 'sex':
            label = (df.sex)
        else:
            label = (df.age)
        return signal, label

def load_raw_data(df, sampling_rate):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data



def load_dataset(list_of_leads):
    sampling_rate=500

    ## According to the data ducomentation it's suggested to use folds 1-8 as training set, fold 9 as validation set and fold 10 as test set.

    # Split data into train, validation and test
    test_fold = 10
    val_fold = 9

    # load and convert annotation data
    Y = pd.read_csv('/home/stu25/project/Age-estimation-and-sex-classification-from-continuous-ECG-PPG/ptbxl_database.csv', index_col='ecg_id')[:500]
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    Y = Y.dropna(subset=['sex', 'age'])

    Y_train = Y[(((Y.strat_fold != test_fold) & (Y.strat_fold != val_fold)))]
    Y_val = Y[(((Y.strat_fold == val_fold)))]
    Y_test = Y[(((Y.strat_fold == test_fold)))]


    dataset_train_sex = ECGDataset(Y_train, 500, 'sex',list_of_leads)
    train_dataloader_sex = DataLoader(dataset_train_sex , batch_size=16, shuffle=True,drop_last=True)
    dataset_train_age = ECGDataset(Y_train, 500, 'age',list_of_leads)
    train_dataloader_age = DataLoader(dataset_train_age , batch_size=64, shuffle=True,drop_last=True)

    dataset_val_sex = ECGDataset(Y_val, 500, 'sex',list_of_leads)
    val_dataloader_sex = DataLoader(dataset_val_sex , batch_size=16, shuffle=False,drop_last=True)
    dataset_val_age = ECGDataset(Y_val, 500, 'age',list_of_leads)
    val_dataloader_age = DataLoader(dataset_val_age , batch_size=64, shuffle=False,drop_last=True)

    dataset_test_sex = ECGDataset(Y_test, 500, 'sex',list_of_leads)
    test_dataloader_sex = DataLoader(dataset_test_sex , batch_size=16, shuffle=False,drop_last=True)
    dataset_test_age = ECGDataset(Y_test, 500, 'age',list_of_leads)
    test_dataloader_age = DataLoader(dataset_test_age , batch_size=64, shuffle=False,drop_last=True)


    return train_dataloader_sex, train_dataloader_age, val_dataloader_sex, val_dataloader_age, test_dataloader_sex, test_dataloader_age
