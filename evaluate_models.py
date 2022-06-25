import torch
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from data_loader import load_dataset
import models
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
import argparse
import os
import pathlib
from sklearn.metrics import r2_score
from sklearn.metrics import multilabel_confusion_matrix

def vals2range(vals):
    vals = vals.copy()
    vals[vals < 18.] = 1
    vals[np.logical_and(18. <= vals,  vals< 30.)] = 2
    vals[np.logical_and(30. <= vals,  vals < 40.)] = 3
    vals[np.logical_and(40 <= vals,  vals < 50)] = 4
    vals[np.logical_and(50 <= vals,  vals < 60)] = 5
    vals[np.logical_and(60 <= vals,  vals < 70)] = 6
    vals[np.logical_and(70 <= vals,  vals < 80)] = 7
    vals[np.logical_and(80 <= vals,  vals < 90)] = 8
    vals[np.logical_and(90 <= vals ,  vals< 100)] = 9
    vals[vals >= 100] = 10

    return vals

def vals2range_paper(vals):
    vals = vals.copy()
    vals[vals <= 18.] = 1
    vals[np.logical_and(18. <= vals,  vals <= 25.)] = 2
    vals[np.logical_and(25. <= vals,  vals <= 50)] = 3
    vals[np.logical_and(50 <= vals,  vals <= 75)] = 4
    vals[vals >= 75] = 5

    return vals

def eval_models(args):
    auc_sex_val = {}
    auc_sex_test = {}
    r2_age_val = {}
    r2_age_test = {}

    r2_age_val_range = {}
    r2_age_test_range = {}

    val_ages_tg = {}
    val_ages_pred = {}

    test_ages_tg = {}
    test_ages_pred = {}

    num=0
    # iterate over saved model
    for saved_model in args.models_path.iterdir():

        if saved_model.name == 'tensor_logs' or saved_model.name[-3:] == 'csv' or saved_model.name == 'CSV_Files':
            continue

        list_of_leads_str = saved_model.name.split('_')[4]
        print(list_of_leads_str)
        list_of_leads = list(map(int, list_of_leads_str[1:-1].split(',')))
        print(list_of_leads)


        _, _, val_dataloader_sex, val_dataloader_age, test_dataloader_sex, test_dataloader_age = load_dataset(
            list_of_leads, args.CSV_file)
        print(saved_model.name[-10:-4])

        if saved_model.name[-10:-4] == 'SexNet':

            print('Sex')
            checkpoint_key = 'sex_net'
            val_dataloader = val_dataloader_sex
            test_dataloader = test_dataloader_sex
            if list_of_leads is None:
                model = models.SexNet()
            else:
                model = models.SexNet(len(list_of_leads))
        elif saved_model.name[-10:-4] == 'AgeNet':
            print('Age')
            val_dataloader = val_dataloader_age
            test_dataloader = test_dataloader_age
            checkpoint_key = 'age_net'
            if list_of_leads is None:
                model = models.AgeNet()
            else:
                model = models.AgeNet(len(list_of_leads))


        checkpoint = torch.load(args.models_path / saved_model.name)
        model.load_state_dict(checkpoint[checkpoint_key])
        model.eval()

        outputs_test_all = []
        label_test_all = []
        outputs_val_all = []
        label_val_all = []

        for i, test_data in enumerate(test_dataloader, 0):
            inputs_test, labels_test = test_data
            inputs_test = inputs_test.permute(0, 2, 1)
            outputs_test = model(inputs_test)
            outputs_test_all.append(outputs_test)
            label_test_all.append(labels_test)

        outputs_test_all = torch.cat(outputs_test_all, dim=0)
        label_test_all = torch.cat(label_test_all, dim=0)

        for i, val_data in enumerate(val_dataloader, 0):
            inputs_val, labels_val = val_data
            inputs_val = inputs_val.permute(0, 2, 1)
            outputs_val = model(inputs_val)
            outputs_val_all.append(outputs_val)
            label_val_all.append(labels_val)

        outputs_val_all = torch.cat(outputs_val_all, dim=0)
        label_val_all = torch.cat(label_val_all, dim=0)

        # plot metrics for Sex
        if saved_model.name[-10:-4] == 'SexNet':
            pred_val = outputs_val_all.detach().numpy()
            tg_val = F.one_hot(label_val_all.to(torch.int64), 2).float().detach().cpu().numpy()

            pred_test = outputs_test_all.detach().numpy()
            tg_test = F.one_hot(label_test_all.to(torch.int64), 2).float().detach().cpu().numpy()

            auc_val = roc_auc_score(tg_val, pred_val)
            auc_test = roc_auc_score(tg_test, pred_test)
            auc_sex_val[list_of_leads_str] = auc_val
            auc_sex_test[list_of_leads_str] = auc_test

            print(f'AUC test:{auc_test}')
            print(f'AUC val:{auc_val}')
            print(auc_sex_val, auc_sex_test)


            fpr_test, tpr_test, _ = metrics.roc_curve(label_test_all.detach().numpy(), outputs_test_all[:, 1].detach().cpu().numpy())
            fpr_val, tpr_val, _ = metrics.roc_curve(label_val_all.detach().numpy(), outputs_val_all[:, 1].detach().cpu().numpy())

            auc_score_val = metrics.auc(fpr_val, tpr_val)
            auc_score_test = metrics.auc(fpr_test, tpr_test)

            plt.title('ROC Curve')
            plt.plot(fpr_val, tpr_val, label='AUC = {:.2f}'.format(auc_score_val))
            plt.plot(fpr_test, tpr_test, label='AUC = {:.2f}'.format(auc_score_test))
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([-0.1, 1.1])
            plt.ylim([-0.1, 1.1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.legend(['val= {:.2f}'.format(auc_score_val), 'test={:.2f}'.format(auc_score_test)], loc='lower right')
            plt.show()

        # plot metrics for age
        if saved_model.name[-10:-4] == 'AgeNet':
            num + 1
            pred_val = outputs_val_all.detach().numpy().squeeze()
            tg_val = label_val_all.detach().numpy()
            pred_test = outputs_test_all.detach().numpy().squeeze()
            tg_test = label_test_all.detach().numpy()

            pred_val_range = vals2range_paper(pred_val)
            tg_val_range = vals2range_paper(tg_val)
            pred_test_range = vals2range_paper(pred_test)
            tg_test_range = vals2range_paper(tg_test)

            print(pred_val_range, tg_val_range)

            val_ages_pred[list_of_leads_str] = pred_val_range
            val_ages_tg[list_of_leads_str] = tg_val_range

            test_ages_tg[list_of_leads_str] = tg_test_range
            test_ages_pred[list_of_leads_str] = pred_test_range

            r2_score_val = r2_score(tg_val, pred_val)
            r2_score_test = r2_score(tg_test, pred_test)
            r2_score_val_range = r2_score(tg_val_range, pred_val_range)
            r2_score_test_range = r2_score(tg_test_range, pred_test_range)

            r2_age_val[list_of_leads_str] = r2_score_val
            r2_age_test[list_of_leads_str] = r2_score_test

            r2_age_val_range[list_of_leads_str] = r2_score_val_range
            r2_age_test_range[list_of_leads_str] = r2_score_test_range


            labels_range = ['under 18', '18 to 25', '25 to 50', '50 to 75', '75 and above']
            cm = confusion_matrix(tg_val_range, pred_val_range)
            sn.heatmap(cm, annot=True, fmt='d', xticklabels=labels_range, yticklabels=labels_range,annot_kws={"size": 15})
            plt.title('Validation',fontsize=20)
            plt.xlabel('Predicted Age', fontsize=15)
            plt.ylabel('Real Age',fontsize=15)
            plt.show()

            cm = confusion_matrix(tg_test_range, pred_test_range)
            sn.heatmap(cm, annot=True, fmt='d', xticklabels=labels_range, yticklabels=labels_range,annot_kws={"size": 15})
            plt.title('Test',fontsize=20)
            plt.xlabel('Predicted Age', fontsize=15)
            plt.ylabel('Real Age', fontsize=15)
            plt.show()

            ages = np.unique(tg_test)
            ages = ages[ages>=18]
            print(ages)
            len_age = len(ages)
            dict_ages = {}
            for i in range(len(ages)//2):
                idx = np.where(tg_test == ages[i])
                pred_temp = pred_test[idx]
                dict_ages[(ages[i])] = pred_temp
            fig, ax = plt.subplots()
            # ax.plot(np.arange(len_age // 2)+1, np.arange(len_age // 2)+1)
            ax.set_xticklabels(dict_ages.keys())
            ind = np.arange(1, len(ages) // 2+2)
            ax.plot(ages[ind])
            ax.boxplot(dict_ages.values())
            plt.show()

            dict_ages = {}
            for i in range(len(ages)//2,len(ages)):
                idx = np.where(tg_test == ages[i])
                pred_temp = pred_test[idx]
                dict_ages[str(ages[i])] = pred_temp
            fig, ax = plt.subplots()
            ax.boxplot(dict_ages.values())
            ax.set_xticklabels(dict_ages.keys())
            ind = np.arange(len(ages) // 2, len(ages))
            ax.plot(ages[ind])
            plt.show()



            plt.figure()
            plt.scatter(tg_val, pred_val, color='orange')
            m_val, b_val = np.polyfit(tg_val, pred_val, 1)
            plt.plot(tg_val, tg_val, color='green',linewidth=3, )
            plt.plot(tg_val, m_val * tg_val + b_val, color='black', linewidth=2,)
            plt.legend(['','Real Age = Estimated Age', 'Linear Regression of Estimated Age' ])
            plt.title('Validation')
            plt.show()

            plt.figure()
            plt.scatter(tg_test, pred_test, color='orange')
            m_test, b_test = np.polyfit(tg_test, pred_test, 1)
            plt.plot(tg_test, tg_test, color='green',linewidth=3,)
            plt.plot(tg_test, m_test * tg_test + b_test, linewidth=3,)
            plt.legend(['','Real Age = Estimated Age', 'Linear Regression of Estimated Age'])
            plt.title('Test')
            plt.show()

            # plt.figure()
            # plt.scatter(tg_test, pred_test)
            # plt.title('Test set')
            # plt.xlabel('real age')
            # plt.ylabel('cnn prediction')
            # plt.show()
            #
            # plt.figure()
            # plt.scatter(tg_val, pred_val)
            # plt.title('Validation set')
            # plt.xlabel('real age')
            # plt.ylabel('cnn prediction')
            # plt.show()
            #
            # plt.figure()
            # plt.scatter(tg_test_range, pred_test_range)
            # plt.title('Test set')
            # plt.xlabel('real age')
            # plt.ylabel('cnn prediction')
            # plt.show()
            #
            # plt.figure()
            # plt.scatter(tg_val_range, pred_val_range)
            # plt.title('Validation set')
            # plt.xlabel('real age')
            # plt.ylabel('cnn prediction')
            # plt.show()
        # if num==1:
        #     break


        df_sex = pd.DataFrame.from_dict(auc_sex_val, 'index', columns=['val'])
        df_sex['test'] = auc_sex_test.values()
        df_sex.to_csv('Sex_auc_results' + str(args.group) +'.csv')

        df_age = pd.DataFrame.from_dict(r2_age_val, 'index', columns=['val'])
        df_age['test'] = r2_age_test.values()
        df_age.to_csv('R2_age_results' + str(args.group) +'.csv')

        df_age_range = pd.DataFrame.from_dict(r2_age_val_range, 'index', columns=['val'])
        df_age_range['test'] = r2_age_test_range.values()
        df_age_range.to_csv('R2_age_range_results' + str(args.group) + '.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', default=None, type=int, nargs="*")
    parser.add_argument('--plot', default=None, type=bool)
    parser.add_argument('--models_path', default=None, type=pathlib.Path)
    parser.add_argument('--CSV_file', default=None, type=str, nargs="*")
    args = parser.parse_args()

    eval_models(args)


