import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import numpy as np

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
    print('正在处理: ',path)
    # 添加患者其他指标信息
    patients_summary = summary[typ]
    patient_info = pd.read_csv(path)
    df_static = pd.DataFrame(columns=static_attribute, index=range(patient_info.shape[0]))
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
    patient_info = patient_info.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
    patient_info.to_csv(save_path)


# tmp_folder = 'tmp_data'
# tmp_files = os.listdir(tmp_folder)

# all_data = []

# for file in tmp_files:
#     if file.endswith('.csv'):
#         patient_data = pd.read_csv(os.path.join(tmp_folder, file))
#         all_data.append(patient_data)

# data = pd.concat(all_data, ignore_index=True)
# data = data.drop(columns=['Date'])

# target_attribute = [
#     '15 min',
#     '30 min',
#     '45 min',
#     '60 min'
# ]


# # 分离特征和目标值
# time_series_features = data[time_serise_attribute].values
# static_features = data[static_attribute].values
# targets = data[target_attribute].values

# def create_sequences(features, targets, static_features, time_steps=10):
#     ts_X, static_X, y = [], [], []
#     for i in range(len(features) - time_steps):
#         ts_X.append(features[i:i+time_steps])
#         static_X.append(static_features[i+time_steps])
#         y.append(targets[i+time_steps])
#     return np.array(ts_X), np.array(static_X), np.array(y)
