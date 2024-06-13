import os
import json
import pandas as pd

url = 'processed-data/Shanghai_T1DM/1001_0_20210730.csv'
df  = pd.read_csv(url)
time_serise_attribute = []
for col in df.columns:
    if col != "Date" and col != 'Unnamed: 0.1' and col != 'Unnamed: 0':
        time_serise_attribute.append(col)
time_serise_attribute_str = json.dumps(time_serise_attribute, indent=4)
with open('time_serise_attribute.json', 'w') as file:
    file.write(time_serise_attribute_str)

df = pd.read_csv('processed-data/Shanghai_T1DM_Summary.csv')
static_attribute = []
for col in df.columns:
    if col != "Patient Number" and col != 'Unnamed: 0.1' and col != 'Unnamed: 0':
        static_attribute.append(col)
static_attribute_str = json.dumps(static_attribute, indent=4)
with open('static_attribute.json', 'w') as file:
    file.write(static_attribute_str)