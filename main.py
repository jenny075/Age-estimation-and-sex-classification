# imports
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import os
from data_loader import load_dataset
import models
import argparse
from tqdm import tqdm
import numpy as np
from itertools import combinations

def writeCSVLoggerFile(csvLoggerFile_path,log):
    df = pd.DataFrame([log])
    # print(df)
    with open(csvLoggerFile_path, 'a') as f:
        df.to_csv(f, mode='a', header=f.tell() == 0, index=False)


def trian(result_dir,args,list_of_leads=None):

    writer = SummaryWriter(result_dir + '/' + 'tensor_logs')
    save_each = 5
    print(result_dir)

    training_log_age = {}
    validaiting_log_age = {}
    training_log_sex = {}
    validaiting_log_sex = {}
    result_dir_csv = result_dir + 'CSV_Files'
    os.makedirs(result_dir_csv, exist_ok=True)
    csvLoggerFile_path_train_age = result_dir_csv + "history_train_age_leads_" + str(list_of_leads) + ".csv"

    csvLoggerFile_path_val_age = result_dir_csv + "history_val_age_leads_" + str(list_of_leads) + ".csv"
    csvLoggerFile_path_train_sex = result_dir_csv + "history_train_sex_leads_" + str(list_of_leads) + ".csv"
    csvLoggerFile_path_val_sex = result_dir_csv + "history_val_sex_leads_" + str(list_of_leads) + ".csv"

    train_dataloader_sex, train_dataloader_age, val_dataloader_sex, val_dataloader_age, test_dataloader_sex, test_dataloader_age = load_dataset(list_of_leads,args.CSV_file)

    if list_of_leads == None:
        age_net = models.AgeNet()
        sex_net = models.SexNet()

    else:
        age_net = models.AgeNet(len(list_of_leads))
        sex_net = models.SexNet(len(list_of_leads))

    criterion_age = nn.MSELoss()
    optimizer_age = optim.Adam(age_net.parameters(), lr=0.0003)

    criterion_sex = nn.BCEWithLogitsLoss()
    optimizer_sex = optim.Adam(sex_net.parameters(), lr=0.0003)

    # train age
    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else 'cpu')
    validation_losses = []
    age_net = age_net.to(device, non_blocking=True)

    for epoch in range(20):  # loop over the dataset multiple times
        print(epoch)
        running_loss = 0.0
        for i, data in enumerate(tqdm(train_dataloader_age, 0)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data  # next(iter(train_dataloader_age))
            #print(inputs.shape)
            # print('Before-',inputs)
            inputs = inputs.to(device, non_blocking=True, dtype=torch.float)
            labels = labels.to(device, non_blocking=True, dtype=torch.float)
            inputs = inputs.permute(0, 2, 1)
            # print('Aftre-',inputs)

            # zero the parameter gradients
            optimizer_age.zero_grad()

            # forward + backward + optimize
            outputs = age_net(inputs)

            loss = criterion_age(outputs, labels)
            if loss.item() is None:
                breakpoint()

            print('batch - {},loss -{}'.format(i, loss.item()))
            loss.backward()
            optimizer_age.step()

            # print statistics
            running_loss += loss.item()
        # if epoch>12:
        #     optimizer_age.param_groups[0]['lr'] = 0.00015

        print(running_loss, i)
        print(f'epoch [{epoch}] - train_avg_loss: {running_loss / (i + 1)}')
        writer.add_scalar("Train/Age_Loss", running_loss / (i + 1), epoch)
        training_log_age["epoch"] = epoch
        training_log_age["ave_loss"] = running_loss / (i + 1)
        writeCSVLoggerFile(csvLoggerFile_path_train_age, training_log_age)
        with torch.no_grad():

            age_net.eval()
            val_loss = 0.0

            for i, val_data in enumerate(tqdm(val_dataloader_age, 0)):
                inputs, labels = val_data
                inputs = inputs.to(device, non_blocking=True, dtype=torch.float)
                labels = labels.to(device, non_blocking=True, dtype=torch.float)
                inputs = inputs.permute(0, 2, 1)
                outputs = age_net(inputs)
                # print(labels, outputs)
                loss = criterion_age(outputs, labels)

                print("val_loss- {} ".format(loss.item()))
                val_loss += loss.item()

            validation_losses.append(val_loss)

        print(f'epoch [{epoch}] - val_avg_loss: {val_loss / (i + 1)}')
        writer.add_scalar("Validation/Age_Loss", val_loss / (i + 1), epoch)
        validaiting_log_age["epoch"] = epoch
        validaiting_log_age["ave_loss"] = val_loss / (i + 1)
        writeCSVLoggerFile(csvLoggerFile_path_val_age, validaiting_log_age)

        if epoch == 0:
            best_loss = val_loss
        print(best_loss)
        # save best model
        if epoch > 0 and validation_losses[-1] < best_loss:
            best_loss = validation_losses[-1]
            print('save best')
            torch.save({
                'age_net': age_net.state_dict(),
                'optimizer_age': optimizer_age.state_dict(),
                'epoch': epoch + 1},
                result_dir+'Best_model_of_leads_'+str(list_of_leads) +'_' +f"AgeNet.pth")


    torch.cuda.empty_cache()
    print('Finished Training')


    validation_losses = []
    sex_net = sex_net.to(device, non_blocking=True)

    for epoch in range(12):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(tqdm(train_dataloader_sex, 0)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device, non_blocking=True, dtype=torch.float)
            labels = labels.to(device, non_blocking=True, dtype=torch.float)
            inputs = inputs.permute(0, 2, 1)

            # zero the parameter gradients
            optimizer_sex.zero_grad()

            # forward + backward + optimize
            outputs = sex_net(inputs)
            loss = criterion_sex(outputs, F.one_hot(labels.to(torch.int64), 2).float())
            print('batch - {},loss -{}'.format(i, loss.item()))
            loss.backward()
            optimizer_sex.step()

            # print statistics
            running_loss += loss.item()

        # if epoch>6:
        #     optimizer_sex.param_groups[0]['lr'] = 0.00015

        print(running_loss, i)
        print(f'epoch [{epoch + 1}] - train_avg_loss: {running_loss / (i + 1)}')
        writer.add_scalar("Train/sex_Loss", running_loss / (i + 1), epoch)
        training_log_sex["epoch"] = epoch
        training_log_sex["ave_loss"] = running_loss / (i + 1)
        writeCSVLoggerFile(csvLoggerFile_path_train_sex, training_log_sex)
        with torch.no_grad():

            sex_net.eval()
            val_loss = 0.0

            for i, val_data in enumerate(tqdm(val_dataloader_sex, 0)):
                inputs, labels = val_data
                inputs = inputs.to(device, non_blocking=True, dtype=torch.float)
                labels = labels.to(device, non_blocking=True, dtype=torch.float)
                inputs = inputs.permute(0, 2, 1)
                outputs = sex_net(inputs)

                loss = criterion_sex(outputs, F.one_hot(labels.to(torch.int64), 2).float())
                print(loss.item())
                val_loss += loss.item()

            validation_losses.append(val_loss)

        print(f'epoch [{epoch}] - val_avg_loss: {val_loss / (i + 1)}')
        writer.add_scalar("Validation/sex_Loss", val_loss / (i + 1), epoch)
        validaiting_log_sex["epoch"] = epoch
        validaiting_log_sex["ave_loss"] = val_loss / (i + 1)
        writeCSVLoggerFile(csvLoggerFile_path_val_sex, validaiting_log_sex)

        if epoch == 0:
            best_loss = val_loss

        # save best model
        if epoch > 0 and validation_losses[-1] < best_loss:
            best_loss = validation_losses[-1]
            print('save best')
            torch.save({
                'sex_net': sex_net.state_dict(),
                'optimizer_sex': optimizer_sex.state_dict(),
                'epoch': epoch + 1},
                result_dir + 'Best_model_of_leads_' + str(list_of_leads) +'_'+ f"SexNet.pth")

            # save each % epoch
        # if epoch > 0 and epoch % save_each == 0:
        #     print('save epoch')
        #     torch.save({
        #         'sex_net': sex_net.state_dict(),
        #         'optimizer_sex': optimizer_sex.state_dict(),
        #         'epoch': epoch + 1},
        #         os.path.join(result_dir, 'Saved_sex', 'Periodic_save', f"SexNet.pth"))

    print('Finished Training')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_of_leads', default=None, type= int, nargs="*")
    parser.add_argument('--group', default=None, type=int)
    parser.add_argument('--title', default=None, type=str)
    parser.add_argument('--alone', default=False, type=bool)
    parser.add_argument('--gpu', default=None, type=str)
    parser.add_argument('--CSV_file', default=None, type=str, nargs="*")

    args = parser.parse_args()

    folder_name = str(args.group)

    result_dir = '/home/stu25/project/new_normed_PYCHARM_spacial_block_'+folder_name+'/'


    if args.alone:
        list_of_leads = np.arange(1,13)
        output = [list(map(list, combinations(list_of_leads, i))) for i in range(len(list_of_leads) + 1)]

        if args.group is not None:
            print(output[args.group])
            for j in range(len(output[args.group])):
                print('Starting Training Leads - ', str(output[args.group][j]))
                trian(result_dir,args, output[args.group][j])
        else:
            print(output)
            for i in range(1,len(output)):
                for j in range(len(output[i])):
                    print('Starting Training Leads - ', str(output[i][j]))
                    trian(result_dir,args, output[i][j])


    else:

        start_time = time.strftime("_%H_%M_%d_%m_%Y_")
        if args.title == None:
            result_dir = result_dir + start_time
        else:
            result_dir = result_dir + args.title
        os.makedirs(result_dir, exist_ok=True)
        if args.list_of_leads is not None:
            if args.group is None:
                print('Recieved the following leads -  ', args.list_of_leads)
                for i in range(len(args.list_of_leads)):
                    print('Starting Training Leads - ',(args.list_of_leads[i]))
                    temp_dir = result_dir +'/Laed_'+str(args.list_of_leads[i])
                    trian(temp_dir,args,[args.list_of_leads[i]])
            else:
                list_of_list = [args.list_of_leads[n:n+args.group] for n in range(0, len(args.list_of_leads), args.group)]
                print('Recieved the following leads -  ',list_of_list)
                for i in range(len(list_of_list)):
                    temp_name = '_'.join(map(str, list_of_list[i]))
                    temp_dir = result_dir +'/Laeds_'+temp_name
                    trian(temp_dir,args,list_of_list[i])
        else:
            trian(result_dir)

