import os
import pandas as pd
from pandas import DataFrame
import json
import re
import time

def read_csv(url) -> DataFrame:
    return pd.read_csv(url)

def save_csv(df: DataFrame, url: str):
    df.to_csv(url)

def drop_yinshi(df: DataFrame) -> DataFrame:
    return df.drop(columns=['饮食'])
    

def basal_insulin_process(df: DataFrame) -> DataFrame:
    attribute_name = 'CSII - basal insulin (Novolin R, IU / H)'
    df_attribute = df[attribute_name]
    # print(df[attribute_name])
    new_df_attribute = df_attribute.copy()
    last_data = 0
    for index, row in enumerate(df_attribute):
        if pd.isna(row):
            new_df_attribute[index] = last_data
        else:
            if row == 'temporarily suspend insulin delivery':
                last_data = 0
                new_df_attribute[index] = last_data
            else:
                last_data = row
    df[attribute_name] = new_df_attribute
    return df

def bolus_insulin_process(df: DataFrame) -> DataFrame:
    attribute_name = 'CSII - bolus insulin (Novolin R, IU)'
    df_attribute = df[attribute_name]
    # print(df[attribute_name])
    new_df_attribute = df_attribute.copy()
    for index, row in enumerate(df_attribute):
        if pd.isna(row):
            new_df_attribute[index] = 0
        else:
            if row == 'temporarily suspend insulin delivery':
                new_df_attribute[index] = 0
    df[attribute_name] = new_df_attribute
    return df

def dietary_intake_process(df: DataFrame) -> DataFrame:
    attribute_name = 'Dietary intake'
    df_attribute = df[attribute_name]
    new_df_attribute = df_attribute.copy()
    for index, row in enumerate(df_attribute):
        if pd.isna(row):
            new_df_attribute[index] = 0
        else:
            new_df_attribute[index] = 1
    df[attribute_name] = new_df_attribute
    # print(new_df_attribute)
    return df

# 皮下注射
def insulin_dose_sc_process(df: DataFrame) -> DataFrame:
    attribute_name = 'Insulin dose - s.c.'
    df_insulin_attribute = []
    with open('insulin_agents.json', 'r') as file:
        agents_map = json.loads(file.read())
    for a in agents_map:
        df_insulin_attribute.append(attribute_name + ' ' + a)
    df_insulin_dose = pd.DataFrame(columns=df_insulin_attribute, index=range(df.shape[0]))
    df_attribute = df[attribute_name]
    for index, row in enumerate(df_attribute):
        for a in df_insulin_attribute:
                df_insulin_dose[a][index] = 0
        if not pd.isna(row):
            agents = row.split(';')
            for _, agent in enumerate(agents):
                searched = False
                for a in agents_map:
                    if re.search(a, agent):
                        ds = re.sub(a, '', agent)
                        ds = re.sub(',', '', ds)
                        ds = re.sub('IU', '', ds)
                        ds = ds.strip()
                        df_insulin_dose[attribute_name + ' ' + a][index] = ds
                        pattren = r'\d+'
                        searched = True
                        if not re.match(pattren, ds):
                            print(attribute_name, '   ', ds, '    ', index)
                if not searched:
                    print(agent, 'sc')


    df = df.drop(columns=[attribute_name])
    df = pd.concat([df, df_insulin_dose], axis=1)
    return df

# 静脉注射
def insulin_dose_iv_process(df: DataFrame) -> DataFrame:
    attribute_name = 'Insulin dose - i.v.'
    df_insulin_attribute = []
    with open('insulin_agents.json', 'r') as file:
        agents_map = json.loads(file.read())
    for a in agents_map:
        df_insulin_attribute.append(attribute_name + ' ' + a)
    df_insulin_attribute.append(attribute_name + ' ' + 'sodium chloride')
    df_insulin_attribute.append(attribute_name + ' ' + 'potassium chloride')
    df_insulin_attribute.append(attribute_name + ' ' + 'glucose')
    df_insulin_dose = pd.DataFrame(columns=df_insulin_attribute, index=range(df.shape[0]))
    df_attribute = df[attribute_name]
    for index, row in enumerate(df_attribute):
        for a in df_insulin_attribute:
            df_insulin_dose[a][index] = 0
        if not pd.isna(row):
            agents = row.split(',')
            for _, agent in enumerate(agents):
                for a in agents_map:
                    if re.search(a, agent):
                        ds = re.sub(a, '', agent)
                        ds = re.sub('IU', '', ds)
                        ds = ds.strip()
                        df_insulin_dose[attribute_name + ' ' + a][index] = ds
                        pattren = r'\d+'
                        if not re.match(pattren, ds):
                            print(attribute_name, '   ', ds, '    ', index)
                if re.search('sodium chloride', agent):
                    ds = agent.strip().split(' ')[0][:-2]
                    # print(ds, agent)
                    df_insulin_dose[attribute_name + ' sodium chloride'][index] = ds
                if re.search('potassium chloride', agent):
                    ds = agent.strip().split(' ')[0]
                    ds = ds.strip()
                    # print(ds, agent)
                    df_insulin_dose[attribute_name + ' sodium chloride'][index] = ds

                if re.search('glucose', agent):
                    if re.search('sodium chloride', agent):
                        ds = agent.strip().split(' ')[0][:-2]
                        # print(ds, agent)
                        df_insulin_dose[attribute_name + ' glucose'][index] = ds
                    else:
                        ds = agent.strip().split(' ')[0]
                        ds = ds.strip()
                        # print(ds, agent)
                        df_insulin_dose[attribute_name + ' glucose'][index] = ds

    df = df.drop(columns=[attribute_name])
    df = pd.concat([df, df_insulin_dose], axis=1)
    return df

# 非胰岛素药物
def non_insulin_hypoglycemic_process(df: DataFrame) -> DataFrame:
    attribute_name = 'Non-insulin hypoglycemic agents'
    df_non_insulin_attribute = []
    with open('non_insulin_agents.json', 'r') as file:
        agents_map = json.loads(file.read())
    for a in agents_map:
        df_non_insulin_attribute.append(attribute_name + ' ' + a)
    df_insulin_dose = pd.DataFrame(columns=df_non_insulin_attribute, index=range(df.shape[0]))
    df_attribute = df[attribute_name]

    
    for index, row in enumerate(df_attribute):
        for a in df_non_insulin_attribute:
            df_insulin_dose[a][index] = 0
        if not pd.isna(row):
            agents = row.split(' ')
            non_insulins = ''
            get_insulins = False
            
            for i, agent in enumerate(agents):
                if i % 3 == 0:
                    for a in agents_map:
                        if re.search(a, agent):
                            non_insulins = a
                            get_insulins = True
                            # print(index, a)
                elif i % 3 == 1:
                    if get_insulins == True:
                        get_insulins = False
                        dose = str(agent.strip())
                        df_insulin_dose[attribute_name + ' ' + non_insulins][index] = dose

                        pattren = r'\d+'
                        if not re.match(pattren, dose):
                            print(attribute_name, '   ', dose, '    ', index)
                    
                else:
                    continue


    df = df.drop(columns=[attribute_name])
    df = pd.concat([df, df_insulin_dose], axis=1)
    return df


def process_CBG_blood_ketone(df:DataFrame) -> DataFrame:
    return df.drop(columns=['CBG (mg / dl)', 'Blood Ketone (mmol / L)'])

def summary_process(df:DataFrame) -> DataFrame:
    # 删除列
    cols_delete = [
        # 'Diabetic Macrovascular Complications',  # 糖尿病大血管并发症
        # 'Acute Diabetic Complications',          # 急性糖尿病并发症
        # 'Diabetic Microvascular Complications',  # 糖尿病微血管并发症
        # 'Comorbidities',                         # 共病
        'Hypoglycemic Agents',                   # 药物
        'Other Agents'                           # 药物
    ]

    df = df.drop(columns=cols_delete)
    df_copy = df.copy()
    
    # 将给列赋值
    for index, item in df_copy.iterrows():
        # Alcohol Drinking History (drinker/non-drinker)
        #   non-drinker
        #   drinker
        if item['Alcohol Drinking History (drinker/non-drinker)'] == 'non-drinker':
            df.loc[index, 'Alcohol Drinking History (drinker/non-drinker)'] = 0
        else:
            df.loc[index, 'Alcohol Drinking History (drinker/non-drinker)'] = 1

        # Type of Diabetes
        #   T1DM
        #   T2DM
        if item['Type of Diabetes'] == 'T1DM':
            df.loc[index, 'Type of Diabetes'] = 0
        else:
            df.loc[index, 'Type of Diabetes'] = 1

        if item['Comorbidities'] == 'none':
            df.loc[index, 'Comorbidities'] = 0
        else:
            df.loc[index, 'Comorbidities'] = 1

        if item['Diabetic Macrovascular  Complications'] == 'none':
            df.loc[index, 'Diabetic Macrovascular  Complications'] = 0
        else:
            df.loc[index, 'Diabetic Macrovascular  Complications'] = 1

        if item['Acute Diabetic Complications'] == 'none':
            df.loc[index, 'Acute Diabetic Complications'] = 0
        else:
            df.loc[index, 'Acute Diabetic Complications'] = 1

        if item['Diabetic Microvascular Complications'] == 'none':
            df.loc[index, 'Diabetic Microvascular Complications'] = 0
        else:
            df.loc[index, 'Diabetic Microvascular Complications'] = 1

        if item['Hypoglycemia (yes/no)'] == 'yes':
            df.loc[index, 'Hypoglycemia (yes/no)'] = 1
        else:
            df.loc[index, 'Hypoglycemia (yes/no)'] = 0

    return df


if __name__ == '__main__':
    T1DM_url = 'raw-data/Shanghai_T1DM'
    T1DM_save = "processed-data/Shanghai_T1DM"
    os.makedirs(T1DM_save, exist_ok=True)
    T1DM_files = os.listdir(T1DM_url)
    for file_url in T1DM_files:
        url = os.path.join(T1DM_url, file_url)
        save_url = os.path.join(T1DM_save, file_url)
        print('正在处理文件：', url)
        df = read_csv(url)
        df = basal_insulin_process(df)
        df = drop_yinshi(df)
        df = dietary_intake_process(df)
        df = insulin_dose_sc_process(df)
        df = non_insulin_hypoglycemic_process(df)
        df = insulin_dose_iv_process(df)
        df = bolus_insulin_process(df)
        df = process_CBG_blood_ketone(df)
        save_csv(df, save_url)

    print('T1DM已处理完毕')

    T2DM_url = 'raw-data/Shanghai_T2DM'
    T2DM_save = "processed-data/Shanghai_T2DM"
    ban_url = [
        '2045_0_20201216.csv', # CSII -bolus insulin没有单位
        '2095_0_20201116.csv' ,# CSII -bolus insulin没有单位
        '2013_0_20220123.csv', # 没有饮食，只有进食量
        '2027_0_20210521.csv', # 胰岛素基础量为中文
    
    ]
    os.makedirs(T2DM_save, exist_ok=True)
    T2DM_files = os.listdir(T2DM_url)
    for file_url in T2DM_files:
        if file_url in ban_url:
            continue
        url = os.path.join(T2DM_url, file_url)
        save_url = os.path.join(T2DM_save, file_url)
        print('正在处理文件：', url)
        df = read_csv(url)
        df = basal_insulin_process(df)
        df = drop_yinshi(df)
        df = dietary_intake_process(df)
        df = insulin_dose_sc_process(df)
        df = non_insulin_hypoglycemic_process(df)
        df = insulin_dose_iv_process(df)
        df = bolus_insulin_process(df)
        df = process_CBG_blood_ketone(df)
        save_csv(df, save_url)

    summarys = ['Shanghai_T1DM_Summary.csv', 'Shanghai_T2DM_Summary.csv']
    with open('static_attribute.json', 'r') as file:
        static_attribute = json.load(file)
    for summary in summarys:
        url = os.path.join('raw-data', summary)
        save_url = os.path.join('processed-data', summary)
        df = read_csv(url)
        df = summary_process(df)
        df.replace('/', float('NaN'), inplace=True)
        for col in static_attribute:
            df[col] = pd.to_numeric(df[col])
            average_value = df[col].mean()
            df[col].fillna(average_value, inplace=True)

        save_csv(df, save_url)




    
    
