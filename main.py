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


def writeCSVLoggerFile(csvLoggerFile_path,log):
    df = pd.DataFrame([log])
    # print(df)
    with open(csvLoggerFile_path, 'a') as f:
        df.to_csv(f, mode='a', header=f.tell() == 0, index=False)


def trian(result_dir,list_of_leads=None):

    writer = SummaryWriter(result_dir + '/' + 'tensor_logs')
    save_each = 5
    print(result_dir)

    training_log_age = {}
    validaiting_log_age = {}
    training_log_sex = {}
    validaiting_log_sex = {}

    if not os.path.isdir(result_dir + '/Saved' + '/Periodic_save'):
        os.makedirs(result_dir + '/Saved' + '/Periodic_save', exist_ok=True)

    if not os.path.isdir(result_dir + '/Saved' + '/Best_model_save'):
        os.makedirs(result_dir + '/Saved' + '/Best_model_save', exist_ok=True)
    csvLoggerFile_path_train_age = os.path.join(result_dir, "history_train_age.csv")
    csvLoggerFile_path_val_age = os.path.join(result_dir, "history_val_age.csv")
    csvLoggerFile_path_train_sex = os.path.join(result_dir, "history_train_sex.csv")
    csvLoggerFile_path_val_sex = os.path.join(result_dir, "history_val_sex.csv")

    train_dataloader_sex, train_dataloader_age, val_dataloader_sex, val_dataloader_age, test_dataloader_sex, test_dataloader_age = load_dataset(list_of_leads)

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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    validation_losses = []
    age_net = age_net.to(device, non_blocking=True)

    for epoch in range(20):  # loop over the dataset multiple times
        print(epoch)
        running_loss = 0.0
        for i, data in enumerate(train_dataloader_age, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data  # next(iter(train_dataloader_age))
            print(inputs.shape)
            # print('Before-',inputs)
            inputs = inputs.to(device, non_blocking=True, dtype=torch.float)
            labels = labels.to(device, non_blocking=True, dtype=torch.float)
            inputs = inputs.permute(0, 2, 1)
            # print('Aftre-',inputs)

            # zero the parameter gradients
            optimizer_age.zero_grad()

            # forward + backward + optimize
            outputs = age_net(inputs)
            # outputs[labels.isnan()]=0
            # labels[labels.isnan()]=0
            loss = criterion_age(outputs, labels)
            print('batch - {},loss -{}'.format(i, loss.item()))
            loss.backward()
            optimizer_age.step()

            # print statistics
            running_loss += loss.item()

        print(running_loss, i)
        print(f'epoch [{epoch}] - train_avg_loss: {running_loss / (i + 1)}')
        writer.add_scalar("Train/Age_Loss", running_loss / (i + 1), epoch)
        training_log_age["epoch"] = epoch
        training_log_age["ave_loss"] = running_loss / (i + 1)
        writeCSVLoggerFile(csvLoggerFile_path_train_age, training_log_age)
        with torch.no_grad():

            age_net.eval()
            val_loss = 0.0

            for i, val_data in enumerate(val_dataloader_age, 0):
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
                os.path.join(result_dir, 'Saved', 'Best_model_save', f"AgeNet.pth"))

            # save each % epoch
        if epoch > 0 and epoch % save_each == 0:
            print('save epoch')
            print(os.path.join(result_dir, 'Saved', 'Periodic_save', f"AgeNet.pth"))
            torch.save({
                'age_net': age_net.state_dict(),
                'optimizer_age': optimizer_age.state_dict(),
                'epoch': epoch + 1},
                os.path.join(result_dir, 'Saved', 'Periodic_save', f"AgeNet.pth"))

    torch.cuda.empty_cache()
    print('Finished Training')

    # train sex

    # start_time = time.strftime("_%d_%m_%Y_%H_%M")
    # result_dir = '/content/drive/MyDrive/results_2' +'/' + start_time
    # os.makedirs(result_dir, exist_ok=True)
    # writer = SummaryWriter(result_dir + '/' + 'tensor_logs')
    # save_each = 5

    if not os.path.isdir(result_dir + '/Saved_sex' + '/Periodic_save'):
        os.makedirs(result_dir + '/Saved_sex' + '/Periodic_save', exist_ok=True)

    if not os.path.isdir(result_dir + '/Saved_sex' + '/Best_model_save'):
        os.makedirs(result_dir + '/Saved_sex' + '/Best_model_save', exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    validation_losses = []
    sex_net = sex_net.to(device, non_blocking=True)

    for epoch in range(12):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_dataloader_sex, 0):
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
        print(running_loss, i)
        print(f'epoch [{epoch + 1}] - train_avg_loss: {running_loss / (i + 1)}')
        writer.add_scalar("Train/sex_Loss", running_loss / (i + 1), epoch)
        training_log_sex["epoch"] = epoch
        training_log_sex["ave_loss"] = running_loss / (i + 1)
        writeCSVLoggerFile(csvLoggerFile_path_train_sex, training_log_sex)
        with torch.no_grad():

            sex_net.eval()
            val_loss = 0.0

            for i, val_data in enumerate(val_dataloader_sex, 0):
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
                os.path.join(result_dir, 'Saved_sex', 'Best_model_save', f"SexNet.pth"))

            # save each % epoch
        if epoch > 0 and epoch % save_each == 0:
            print('save epoch')
            torch.save({
                'sex_net': sex_net.state_dict(),
                'optimizer_sex': optimizer_sex.state_dict(),
                'epoch': epoch + 1},
                os.path.join(result_dir, 'Saved_sex', 'Periodic_save', f"SexNet.pth"))

    print('Finished Training')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_of_leads', default=None, type= int, nargs="*")
    parser.add_argument('--group', default=None, type=int)
    parser.add_argument('--title', default=None, type=str)
    args = parser.parse_args()

    start_time = time.strftime("_%H_%M_%d_%m_%Y_")
    result_dir = '/home/stu25/project/results_2' + '/' +args.title + start_time
    os.makedirs(result_dir, exist_ok=True)
    if args.list_of_leads is not None:
        if args.group is None:
            print('Recieved the following leads -  ', args.list_of_leads)
            for i in range(len(args.list_of_leads)):
                temp_dir = result_dir +'/Laed_'+str(args.list_of_leads[i])
                trian(temp_dir,[args.list_of_leads[i]])
        else:
            list_of_list = [args.list_of_leads[n:n+args.group] for n in range(0, len(args.list_of_leads), args.group)]
            print('Recieved the following leads -  ',list_of_list)
            for i in range(len(list_of_list)):
                temp_name = '_'.join(map(str, list_of_list[i]))
                temp_dir = result_dir +'/Laeds_'+temp_name
                trian(temp_dir,list_of_list[i])
    else:
        trian(result_dir)

