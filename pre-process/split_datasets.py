import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json

T1DM_folder = 'processed-data/Shanghai_T1DM'
T2DM_folder = 'processed-data/Shanghai_T2DM'
save_folder = 'tmp_data'


summary = {
    'Shanghai_T1DM': pd.read_csv('processed-data/Shanghai_T1DM_Summary.csv'),
    'Shanghai_T2DM': pd.read_csv('processed-data/Shanghai_T2DM_Summary.csv')
}

T1DM = os.listdir(T1DM_folder)
T2DM = os.listdir(T2DM_folder)
files_path = []

for i in T1DM:
    files_path.append(os.path.join(T1DM_folder, i))


for i in T2DM:
    files_path.append(os.path.join(T2DM_folder, i))





patients = []
for file_path in files_path:
    file_name = file_path.split('/')[-1]
    type_t = file_path.split('/')[-2]
    patient = file_name.split('.')[0]
    patients.append((type_t, file_path, patient))


with open('static_attribute.json', 'r') as file:
    static_attribute = json.load(file)

with open('time_serise_attribute.json', 'r') as file:
    time_serise_attribute = json.load(file)

for typ, path, patient in patients:
    # 添加患者其他指标信息
    patients_summary = summary[typ]
    patient_info = pd.read_csv(path)
    df_static = pd.DataFrame(columns=time_serise_attribute, index=range(patient_info.shape[0]))
    patient_static = patients_summary[patients_summary['Patient Number'] == patient]
    for a in static_attribute:
        c = patient_static[a].values[0]
        df_static[a] = c
    patient_info = pd.concat([patient_info, df_static], axis=1)
    
    # 添加target值
    patient_info['15 min'] = patient_info['CGM (mg / dl)'].shift(-1)
    patient_info['30 min'] = patient_info['CGM (mg / dl)'].shift(-2)
    patient_info['45 min'] = patient_info['CGM (mg / dl)'].shift(-3)
    patient_info['60 min'] = patient_info['CGM (mg / dl)'].shift(-4)

    # 处理数据，处理掉没有target值的行，让数据量可以被10整除
    patient_info = patient_info.dropna()

    rows_to_keep = len(patient_info) - (len(patient_info) % 10)

    patient_info = patient_info[:rows_to_keep]

    save_path = os.path.join(save_folder, patient + '.csv')
    patient_info.to_csv(save_path)