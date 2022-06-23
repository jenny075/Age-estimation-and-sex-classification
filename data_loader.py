import pandas as pd
import numpy as np
import wfdb
import ast
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import glob
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

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
        #
        # if self.sampling_rate == 100:
        #     data = wfdb.rdsamp('/home/stu25/project/data/' + df.filename_lr)
        data = wfdb.rdsamp(df.path_file)
        signal, _ = data
        signal = (signal - np.min(signal, axis=0)) / (np.max(signal, axis=0) - np.min(signal, axis=0))
        data = np.array(signal)

        if self.list_of_leads == None:
            data_pad = np.zeros((data.shape[0] + 120, data.shape[1]))
            data_pad[60:-60, :] = data
        else:
            data_pad = np.zeros((data.shape[0] + 120, len(self.list_of_leads)))
            data_pad[60:-60, :] = data[:,np.array(self.list_of_leads)-1]
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



def load_dataset(list_of_leads,csv_file = None):
    sampling_rate=500

    ## According to the data ducomentation it's suggested to use folds 1-8 as training set, fold 9 as validation set and fold 10 as test set.

    # Split data into train, validation and test
    test_fold = 10
    val_fold = 9

    # load and convert annotation data
    if csv_file is None:
        Y = pd.read_csv('/home/stu25/project/Age-estimation-and-sex-classification-from-continuous-ECG-PPG/ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        Y = Y.dropna(subset=['sex', 'age'])

        Y_train = Y[(((Y.strat_fold != test_fold) & (Y.strat_fold != val_fold)))]
        Y_val = Y[(((Y.strat_fold == val_fold)))]
        Y_test = Y[(((Y.strat_fold == test_fold)))]

    else:
        extension = 'csv'
        all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
        # combine all files in the list
        Y = pd.concat([pd.read_csv(f) for f in csv_file])
        #Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        Y = Y.dropna(subset=['sex', 'age'])
        Y.to_csv("combined_csv.csv", index=False, encoding='utf-8-sig')
        Y = Y.assign(fold='train')
        Y_f = Y[Y['sex']=='Female']
        Y_f['sex'] == 1
        Y_m = Y[Y['sex'] == 'Male']
        Y_m['sex'] == 0
        count = 0
        hist_count, bins, patch = plt.hist(Y_f['age'])
        new_df = pd.DataFrame(columns=Y_f.columns)
        for i in range(len(hist_count)):
            Y_temp = Y_f[count:int(hist_count[i])+count]
            Y_temp = shuffle(Y_temp)
            Y_temp.iloc[0:np.round(0.1 * len(Y_temp)).astype('int'), -1] = 'test'
            Y_temp.iloc[np.round(0.1 * len(Y_temp)).astype('int'):np.round(0.3 * len(Y_temp)).astype('int'), -1] = 'val'
            new_df = new_df.append(Y_temp, ignore_index=True)
            count = count + int(hist_count[i])
        Y_f = new_df
        count = 0
        hist_count, bins, patch = plt.hist(Y_m['age'])
        new_df = pd.DataFrame(columns=Y_m.columns)
        for i in range(len(hist_count)):
            Y_temp = Y_m[count:int(hist_count[i])+count]
            Y_temp = shuffle(Y_temp)
            Y_temp.iloc[0:np.round(0.1 * len(Y_temp)).astype('int'), -1] = 'test'
            Y_temp.iloc[np.round(0.1 * len(Y_temp)).astype('int'):np.round(0.3 * len(Y_temp)).astype('int'), -1] = 'val'
            new_df = new_df.append(Y_temp, ignore_index=True)
            count = count + int(hist_count[i])
        Y_m = new_df
        Y_new = Y_f.append(Y_m, ignore_index=True)
        Y_train = Y_new[(Y_new.fold == 'train')]
        Y_val = Y_new[(Y_new.fold == 'val')]
        Y_test = Y_new[Y_new.fold == 'test']





    # Y_sex_train = Y_train.sex
    # Y_age_train = Y_train.age
    #
    # Y_sex_val = Y_val.sex
    # Y_age_val = Y_val.age
    #
    # Y_sex_test = Y_test.sex
    # Y_age_test = Y_test.age
    # #
    # fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(9, 12))
    # img1 = axs[0, 0].hist(Y_sex_train)
    # axs[0, 0].set_title('Y_sex train')
    # img2 = axs[0, 1].hist(Y_age_train)
    # axs[0, 1].set_title('Y_age train')
    # img3 = axs[1, 0].hist(Y_sex_val)
    # axs[1, 0].set_title('Y_sex val')
    # img4 = axs[1, 1].hist(Y_age_val)
    # axs[1, 1].set_title('Y_age val')
    # img3 = axs[2, 0].hist(Y_sex_test)
    # axs[2, 0].set_title('Y_sex test')
    # img4 = axs[2, 1].hist(Y_age_test)
    # axs[2, 1].set_title('Y_age test')
    # plt.show()

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
