import os
import pandas as pd
from pandas import DataFrame
import json
import re

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
    last_data = None
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
    with open('agents_info.json', 'r') as file:
        agents_map = json.loads(file.read())
    df_attribute = df[attribute_name]
    new_attribute_1 = df_attribute.copy()
    new_attribute_2 = df_attribute.copy()
    for index, row in enumerate(df_attribute):
        # print(row)
        if pd.isna(row):
            new_attribute_1[index] = "-1"
            new_attribute_2[index] = "0"
        else:
            agents = row.split(';')
            # print(index, row, agents)
            insulins = []
            dose = []
            for i, agent in enumerate(agents):
                # if i % 2 == 0:
                #     t = re.sub('\xa0', ' ', agent.strip())
                #     insulins.append(str(agents_map[t]))
                # else:
                #     dose.append(str(agent.strip()))
                for a in agents_map:
                    if re.search(a, agent):
                        insulins.append(str(agents_map[a]))
                        ds = re.sub(a, '', agent)
                        ds = re.sub(',', '', ds)
                        ds = re.sub('IU', '', ds)
                        dose.append(ds.strip())
            new_attribute_1[index] = ','.join(insulins)
            new_attribute_2[index] = ','.join(dose)


    df['Insulin kind - s.c.'] = new_attribute_1
    df['Dose - s.c.'] = new_attribute_2
    df = df.drop(columns=['Insulin dose - s.c.'])
    return df

# 静脉注射
def insulin_dose_iv_process(df: DataFrame) -> DataFrame:
    attribute_name = 'Insulin dose - i.v.'
    with open('agents_info.json', 'r') as file:
        agents_map = json.loads(file.read())
    df_attribute = df[attribute_name]
    new_attribute_1 = df_attribute.copy()
    new_attribute_2 = df_attribute.copy()
    for index, row in enumerate(df_attribute):
        # print(row)
        if pd.isna(row):
            new_attribute_1[index] = "-1"
            new_attribute_2[index] = "0"
        else:
            agents = row.split(',')
            insulins = []
            dose = []
            for _, agent in enumerate(agents):
                for a in agents_map:
                    if re.search(a, agent):
                        insulins.append(str(agents_map[a]))
                        ds = re.sub(a, '', agent)
                        ds = re.sub('IU', '', ds)
                        dose.append(ds.strip())
            # if insulins != []:
            new_attribute_1[index] = ','.join(insulins)
            new_attribute_2[index] = ','.join(dose)


    df['Insulin kind - i.v.'] = new_attribute_1
    df['Dose - i.v.'] = new_attribute_2
    df = df.drop(columns=[attribute_name])
    return df



# 非胰岛素药物
def non_insulin_hypoglycemic_process(df: DataFrame) -> DataFrame:
    attribute_name = 'Non-insulin hypoglycemic agents'
    with open('agents_info.json', 'r') as file:
        agents_map = json.loads(file.read())
    df_attribute = df[attribute_name]
    new_attribute_1 = df_attribute.copy()
    new_attribute_2 = df_attribute.copy()
    for index, row in enumerate(df_attribute):
        # if index == 861:
        #     print(row)
        if pd.isna(row):
            new_attribute_1[index] = "-1"
            new_attribute_2[index] = "0"
        else:
            agents = row.split(' ')
            insulins = []
            dose = []
            for i, agent in enumerate(agents):
                if i % 3 == 0:
                    for a in agents_map:
                        if re.search(a, agent):
                            insulins.append(a)
                elif i % 3 == 1:
                    dose.append(str(agent.strip()))
                else:
                    continue
            new_attribute_1[index] = ','.join(insulins)
            new_attribute_2[index] = ','.join(dose)


    df['Non-insulin hypoglycemic agents'] = new_attribute_1
    df['Non-insulin hypoglycemic agents dose'] = new_attribute_2
    # print(new_attribute_2[861])
    # print(new_attribute_1[861])
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
        save_csv(df, save_url)

    summarys = ['Shanghai_T1DM_Summary.csv', 'Shanghai_T2DM_Summary.csv']
    for summary in summarys:
        url = os.path.join('raw-data', summary)
        save_url = os.path.join('processed-data', summary)
        df = read_csv(url)
        save_csv(df, save_url)
    
    #save_csv(df, save_url)


            

