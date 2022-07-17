import pathlib
import wfdb
import pandas as pd

print("Starting")
data_path = pathlib.Path("/home/stu25/project_2/new_data/Germany/")
df = pd.DataFrame(columns=['patient_id','path_file' ,'age', 'sex'])
i=1
for data in data_path.iterdir():
    data_name = str(data.name[:-4])
    ending = str(data.name[-3:])
    if "hea" not in ending:
      continue    
    data = wfdb.rdsamp(str(data_path/data_name))
    path = str(data_path/data_name)
    age = data[1]['comments'][0]
    sex = data[1]['comments'][1]
    age = age.split(' ')[1]
    sex = sex.split(' ')[1]
    df = pd.concat([df,pd.Series([data_name,path, age, sex],index=['patient_id','path_file' ,'age', 'sex']).to_frame().T],
                   ignore_index=True)
df.to_csv('data_labels_Germany.csv')
print("Finished")
