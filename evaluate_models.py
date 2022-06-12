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


_,_ , val_dataloader_sex, val_dataloader_age, test_dataloader_sex, test_dataloader_age =  load_dataset()

print('val_sex:',len(val_dataloader_sex), 'test_sex:',len(test_dataloader_sex),'val_age:',len(val_dataloader_age),'val_age:',len(test_dataloader_age))


age_net = models.AgeNet()
sex_net = models.SexNet()

best_model_sex = "/home/stu25/project/results_2/_12_06_2022_12_04/Saved_sex/Best_model_save/SexNet.pth"
best_model_age = "/home/stu25/project/results_2/_12_06_2022_12_04/Saved/Best_model_save/AgeNet.pth"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
'''
# Eval Sex 
checkpoint = torch.load(best_model_sex)
sex_net.load_state_dict(checkpoint['sex_net'])

sex_net = sex_net.to(device, non_blocking=True)
sex_net.eval()
outputs_test_all = []
label_test_all = []
outputs_val_all = []
label_val_all = []
for i, test_data in enumerate(test_dataloader_sex,0):

            inputs_test, labels_test = test_data
            inputs_test = inputs_test.to(device, non_blocking=True, dtype=torch.float)
            labels_test = labels_test.to(device, non_blocking=True, dtype=torch.float)
            inputs_test = inputs_test.permute(0,2,1)
            outputs_test = sex_net(inputs_test)
            outputs_test_all.append(outputs_test)
            label_test_all.append(labels_test)

outputs_test_all = torch.cat(outputs_test_all,dim = 0)
label_test_all = torch.cat(label_test_all,dim = 0)

pred_test_sex = outputs_test_all.detach().cpu().numpy()
y_test_sex_new = F.one_hot(label_test_all.to(torch.int64),2).float().detach().cpu().numpy()
auc_test = roc_auc_score(y_test_sex_new, pred_test_sex)
torch.cuda.empty_cache()


for i, val_data in enumerate(val_dataloader_sex,0):

            inputs_val, labels_val = val_data
            inputs_val = inputs_val.to(device, non_blocking=True, dtype=torch.float)
            labels_val = labels_val.to(device, non_blocking=True, dtype=torch.float)
            inputs_val = inputs_val.permute(0,2,1)
            outputs_val = sex_net(inputs_val)
            outputs_val_all.append(outputs_val)
            label_val_all.append(labels_val)

outputs_val_all = torch.cat(outputs_val_all,dim = 0)
label_val_all = torch.cat(label_val_all,dim = 0)

pred_val_sex = outputs_val_all.detach().cpu().numpy()
y_val_sex_new = F.one_hot(label_val_all.to(torch.int64),2).float().detach().cpu().numpy()
auc_val = roc_auc_score(y_val_sex_new, pred_val_sex)


print(f'AUC test:{auc_test}')
print(f'AUC val:{auc_val}')


fpr, tpr, _ = metrics.roc_curve(label_test_all.detach().cpu().numpy(), outputs_test_all[:,1].detach().cpu().numpy())

auc_score = metrics.auc(fpr, tpr)

# clear current figure


plt.title('ROC Curve')
plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(auc_score))

# it's helpful to add a diagonal to indicate where chance 
# scores lie (i.e. just flipping a coin)
plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.legend(loc='lower right')
plt.show()
torch.cuda.empty_cache()
'''
# Eval Age 

checkpoint = torch.load(best_model_age)
age_net.load_state_dict(checkpoint['age_net'])
age_net = age_net.to(device, non_blocking=True)
age_net.eval()

outputs_test_all = []
label_test_all = []
outputs_val_all = []
label_val_all = []


for i, test_data in enumerate(test_dataloader_age,0):

            inputs_test, labels_test = test_data
            inputs_test = inputs_test.to(device, non_blocking=True, dtype=torch.float)
            labels_test = labels_test.to(device, non_blocking=True, dtype=torch.float)
            inputs_test = inputs_test.permute(0,2,1)
            outputs_test = age_net(inputs_test)
            outputs_test_all.append(outputs_test)
            label_test_all.append(labels_test)

outputs_test_all = torch.cat(outputs_test_all,dim = 0)
label_test_all = torch.cat(label_test_all,dim = 0).detach().cpu().numpy()

pred_test_age = outputs_test_all.detach().cpu().numpy()
print(type(outputs_test_all), type(pred_test_age))
cm = confusion_matrix(outputs_test_all.detach().cpu().numpy(), pred_test_age)
f = sn.heatmap(cm, annot=True, fmt='d')

plt.figure()
plt.scatter(label_test_all, pred_test_age.squeeze())
plt.title('Test set')
plt.xlabel('real age')
plt.ylabel('cnn prediction')
plt.show()

for i, val_data in enumerate(val_dataloader_age,0):

            inputs_val, labels_val = val_data
            inputs_val = inputs_val.to(device, non_blocking=True, dtype=torch.float)
            labels_val = labels_val.to(device, non_blocking=True, dtype=torch.float)
            inputs_val = inputs_val.permute(0,2,1)
            outputs_val = age_net(inputs_val)
            outputs_val_all.append(outputs_val)
            label_val_all.append(labels_val)

outputs_val_all = torch.cat(outputs_val_all,dim = 0)


label_val_all = torch.cat(label_val_all,dim = 0).detach().cpu().numpy()
pred_val_age = outputs_val_all.detach().cpu().numpy()
print(type(label_val_all), type(pred_val_age))
cm = confusion_matrix(label_val_all.detach().cpu().numpy(), pred_val_age)
f = sn.heatmap(cm, annot=True, fmt='d')

plt.figure()
plt.scatter(label_val_all, pred_val_age)
plt.title('Validation set')
plt.xlabel('real age')
plt.ylabel('cnn prediction')
plt.show()
torch.cuda.empty_cache()
