# Age Estimation And Sex Classification From Continuous ECG

This repo provides the code for the work of age estimation and sex classification from  reduced number of ECG leads.
We used and modified the CNN architectures presented in 2019 this [paper](https://www.ahajournals.org/doi/10.1161/CIRCEP.119.007284) 
by Attia et al. We reffer to the networks as SexNet and AgeNet respectivly.   

### Data sets

For the models training and evaluation we used the following 3 publicly available datasets 
Chapman-Shaoxing, Ningbo First Hospital and PBL-XL. In total we had 66,746 samples, 55 percents were males and 45 
percents females with various range of ages from 18 to 95.
The dataset was split into train validation and test sets with a the ration of 70:20:10, 
with similar distribution of age and sex in each set.

### Expirements Results

we trained the AgeNet and SexNet on the 12-leads ECG inputs to create baseline models and reproduce the paper results.
Next, we trained the networks on different combinations of a reduced set of leads
to examine our hypothesis. The models were trained on 4-
leads, 6-leads, and 8-leads inpu


| Model      | AgeNet (R^2) |  SexNet (AUC)   | 
| :--------- | :------:     |  :----: |
| papper     | 0.7  | 0.97 | 
| 12-leads 3 datasets |  0.4851 | 0.8955  | 
| 12-leads 2 datasets         | 0.2355  |  0.8784 |
| 8-leads 3 datasets        | 0.4804  | 0.8915 | 
| 6-leads 3 datasets       |  0.4771  |  0.8840 |  
| 4-leads 3 datasets       |  0.4462 |  0.8718 |  

### Code Usage

*`main.py` - Main training script for AgeNet and SexNet. 

*`evaluate_models.py` - Evalution and plotting of the preformance of different models.

