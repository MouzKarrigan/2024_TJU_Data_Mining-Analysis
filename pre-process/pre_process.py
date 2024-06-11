import os
import pandas as pd
from pandas import DataFrame
import json

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
            last_data = row
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
    print(new_df_attribute)
    return df

# 皮下注射
def insulin_dose_sc_process(df: DataFrame) -> DataFrame:
    agents_map = json.loads('agents_info.json')
    




# 静脉注射



# 非胰岛素药物

if __name__ == '__main__':
    url = 'datasets/Shanghai_T1DM/1003_0_20210831.csv'
    # basal_insulin_process(url)
    df = read_csv(url)
    df = basal_insulin_process(df)
    df = drop_yinshi(df)
    df = dietary_intake_process(df)
    # 其他处理
    
    #save_csv(df, save_url)


            

