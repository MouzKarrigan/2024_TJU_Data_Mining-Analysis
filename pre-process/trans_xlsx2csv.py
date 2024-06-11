import pandas as pd
import os
# import openpyxl
base_url = 'diabetes_datasets'
new_folder = 'datasets'
new_T1DM_folder_url = os.path.join(new_folder, "Shanghai_T1DM")
T1DM_folder_url = os.path.join(base_url, 'Shanghai_T1DM')
T1DM_summary_url = os.path.join(base_url, "Shanghai_T1DM_Summary.xlsx")
new_T2DM_folder_url = os.path.join(new_folder, "Shanghai_T2DM")
T2DM_folder_url = os.path.join(base_url, 'Shanghai_T2DM')
T2DM_summary_url = os.path.join(base_url, "Shanghai_T2DM_Summary.xlsx")

os.makedirs(new_folder, exist_ok=True)
os.makedirs(new_T1DM_folder_url, exist_ok=True)
os.makedirs(new_T2DM_folder_url, exist_ok=True)

T1DM = os.listdir(T1DM_folder_url)
T1DM_summary = pd.read_excel(T1DM_summary_url)
T1DM_summary.to_csv(os.path.join(new_folder, "Shanghai_T1DM_Summary.csv"))

T2DM = os.listdir(T2DM_folder_url)
T2DM_summary = pd.read_excel(T2DM_summary_url)
T2DM_summary.to_csv(os.path.join(new_folder, "Shanghai_T2DM_Summary.csv"))


# patients = []

# for id in T1DM_summary["Patient Number"]:
#     patients.append(id)

for file in T1DM:
    if file == '.DS_Store':
        continue
    data = pd.read_excel(os.path.join(T1DM_folder_url, file))

    new_filename = file.split('.')[0] + '.csv'
    data.to_csv(os.path.join(new_T1DM_folder_url, new_filename))

for file in T2DM:
    if file == '.DS_Store':
        continue
    data = pd.read_excel(os.path.join(T2DM_folder_url, file))

    new_filename = file.split('.')[0] + '.csv'
    data.to_csv(os.path.join(new_T2DM_folder_url, new_filename))