# 2024_TJU_Data_Mining-Analysis_Report

# 1 问题分析

## 1.1 问题概述

## 1.2 当前主流血糖预测方法

## 1.3 

## 1.4 小组人员构成

2151617 郑埴 2152970 李锦霖 2154306 李泽凯 2154314 郑楷

# 2 数据集分析

## 2.1 所用数据集介绍

## 2.2 数据预处理

### 2.2.1 格式转换

为方便后续处理，首先将T1DM与T2DM数据集中所有.xlsx文件转化为.csv文件并存储。

```python
for file in T1DM:
    if file == '.DS_Store':
        continue
    data = pd.read_excel(os.path.join(T1DM_folder_url, file))

    new_filename = file.split('.')[0] + '.csv'
    data.to_csv(os.path.join(new_T1DM_folder_url, new_filename))
```

### 2.2.2 Dietary Intake Pre-process

在原数据集中，有两个Attribute分别表示患者`Dietary Intake`的内容及其英文翻译。我们将这两个Attribute合二为一，简单判断Dietary Intake原数据中是否有内容，若有内容则表示患者此时进食，在输出数据的`Dietary Intake`中标记为`1`，若无进食则在输出数据中标记为`0`。

```python
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
```

我们忽略进食的具体内容，只考虑患者进食与否，目的是简化后续的模型构建与运算。在数据量较少的前提下，分析患者进食对血糖水平的平均影响，远比分析患者每次进食的具体营养物质对血糖水平的不同影响，来得准确且便捷。

### 2.2.3 Agent Details Pre-process

在原数据集中的`Shanghai_T1DM_Summary.csv`与`Shanghai_T2DM_Summary.csv`里介绍了数据集中所有所用的药物名称与其是否含胰岛素的信息。为了方便后续的数据预处理，我们将所有的药物名称通过`agents_process.py`识别并提取到`agents_info.json`中。

```python
urls = ['Shanghai_T1DM_Summary.csv', 'Shanghai_T2DM_Summary.csv']
agents = set()
agents_list = []

for url in urls:
    full_url = os.path.join('raw-data', url)
    df = pd.read_csv(full_url)
    df_agents = df["Hypoglycemic Agents"]
    for i in df_agents:
        some_agents = i.split(',')
        for j in some_agents:
            if j != 'none':
                agents.add(j.strip())


for agent in agents:
    agents_list.append(agent)

agents_list_sorted = sorted(agents_list, key=len)

agent_str = json.dumps(agents_list, indent=2)

with open('agents_info.json', 'w') as file:
    file.write(agent_str)
```
之后将含胰岛素的降糖药物名称提取并保存至`insulin_agents.json`，将不含胰岛素的降糖药物名称提取并保存至`non_insulin_agents.json`。

### 2.2.4 Insulin Dose - s.c. Pre-process

在原数据集中有Attribute`Insulin Dose - s.c.`以类似`Novolin R, 2 IU`的形式记录患者通过皮下注射的药物名称与剂量，逗号前是药物名称，逗号后是注射剂量。为此我们设计了方法分别提取`Insulin Dose - s.c.`这一Attribute中所记录的药物名称与剂量，并记录在输出数据中。

```python
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
                for a in agents_map:
                    if re.search(a, agent):
                        ds = re.sub(a, '', agent)
                        ds = re.sub(',', '', ds)
                        ds = re.sub('IU', '', ds)
                        ds = ds.strip()
                        df_insulin_dose[attribute_name + ' ' + a][index] = ds
                        pattren = r'\d+'
                        if not re.match(pattren, ds):
                            print(ds) 


    df = df.drop(columns=[attribute_name])
    df = pd.concat([df, df_insulin_dose], axis=1)
    return df
```

值得注意的是，由于单次注射不一定注射单一种类药物，而可能同时注射两种甚至更多的不同种类药物，因此我们在预处理后的输出数据中，不采用将`Insulin Dose - s.c.`这一Attribute转化为`Insulin Kind`和`Dose`两种Attribute的常规方法，而是把`insulin_agents.json`中所有出现过的含胰岛素的降糖药物名称都分别设置为新属性，在输出数据中仅在对应药物种类属性下记录剂量，单位为IU，若无此种药物则剂量记为`0`。如此便可处理两种甚至更多的不同种类药物同时注射的情况。

### 2.2.5 Insulin Dose - i.v. Pre-process

在原数据集中有Attribute`Insulin Dose - s.c.`以类似`500ml 0.9% sodium chloride,  12 IU Novolin R,  10 ml 10% potassium chloride`的形式记录患者通过静脉注射的药物名称与剂量。我们忽略前后两种对血糖无影响的辅药成分，仅提取每条记录中间对于血糖浓度有影响的降糖药物主要成分。

```python
def insulin_dose_iv_process(df: DataFrame) -> DataFrame:
    attribute_name = 'Insulin dose - i.v.'
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
                            print(ds)     


    df = df.drop(columns=[attribute_name])
    df = pd.concat([df, df_insulin_dose], axis=1)
    return df
```

处理过程与2.2.4类似，把`insulin_agents.json`中所有出现过的含胰岛素的降糖药物名称都分别设置为新属性，在输出数据中仅在对应药物种类属性下记录剂量，单位为IU，若无此种药物则剂量记为`0`。值得注意的是，为了将静脉注射的药物与皮下注射的药物区分开，新添加的各种药物剂量属性中会根据注射形式的不同而有`Insulin dose - s.c.`或`Insulin dose - i.v.`前缀，以便后续模型计算的区分。

### 2.2.6 Non-insulin Hypoglycemic Agents Pre-process

在原数据集中有Attribute`Non-insulin hypoglycemic agents`记录患者摄入非胰岛素降糖药物的种类与剂量，我们对该Attribute的预处理与2.2.4和2.2.5类似。

```python
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
                elif i % 3 == 1:
                    if get_insulins == True:
                        dose = str(agent.strip())
                        try:
                            df_insulin_dose[attribute_name + ' ' + non_insulins][index] = dose
                        except UnboundLocalError as e:
                            print(attribute_name + ' ' + insulins)
                            exit(-1)

                        pattren = r'\d+'
                        if not re.match(pattren, dose):
                            print(dose)
                    
                else:
                    insulins = ''
                    continue


    df = df.drop(columns=[attribute_name])
    df = pd.concat([df, df_insulin_dose], axis=1)
    return df
```

所有在输出数据中添加的新属性由前缀`Non-insulin hypoglycemic agents`加上药物名称构成，从而与2.2.4和2.2.5过程中添加的新属性区分开，属性中只记录该种药物的剂量。

### 2.2.7 CSII Pre-process

在原数据集中的Attribute`CSII - bolus insulin (Novolin R, IU)`与`CSII - basal insulin (Novolin R, IU / H)`，后者在.xlsx文件中是以合并单元格的形式表现该时段内的持续基底胰岛素剂量，转化为.csv后出现了一定程度的数据丢失，我们设计了方法将其补全，同样记录在输出数据中一整个时段的`CSII - basal insulin (Novolin R, IU / H)`中，但是是逐条分开记录而非合并记录。

```python
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
```

值得注意的是在`CSII - bolus insulin (Novolin R, IU)`与`CSII - basal insulin (Novolin R, IU / H)`两个Attribute中会有`temporarily suspend insulin delivery`记录出现，表示在下一小段时间内停止了持续给药，我们设计了方法识别并在输出数据中将之后时段的剂量记录为`0`直到有新的剂量给出。

```python
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
```

### 2.2.8 Other Attributes Pre-process

在原数据中给出的其他Attributes中，`Date`与`CGM (mg / dl)`可以直接使用，因此我们将其转移到输出数据中。

```python
def read_csv(url) -> DataFrame:
    return pd.read_csv(url)

def save_csv(df: DataFrame, url: str):
    df.to_csv(url)
```

而Attribute`CBG (mg / dl)`指的是采用另一种方式测得的血糖浓度，空值较多难以在模型中处理，且与`CGM (mg / dl)`相重复，因此我们在输出数据中删去了`CBG (mg / dl)`这一Attribute。`Blood Ketone (mmol / L)`表示血酮含量与本问题相关性较差，同时空值也较多难以处理，因此该Attribute也被我们在输出数据中删去。

```python
def process_CBG_blood_ketone(df:DataFrame) -> DataFrame:
    return df.drop(columns=['CBG (mg / dl)', 'Blood Ketone (mmol / L)'])
```

### 2.2.8 Data Select

在原数据集中的若干.xlsx里，有几份表格因为研究人员的疏忽在记录时出现了严重的格式错误，由于只有极少部分的数据受到影响，因此本着开发成本上的考量，我们不再额外设计方法处理这类格式出现严重问题的数据，而是直接将它们排除。具体理由如下。

```python
ban_url = [
        '2045_0_20201216.csv', # CSII -bolus insulin没有单位
        '2095_0_20201116.csv' ,# CSII -bolus insulin没有单位
        '2013_0_20220123.csv', # 没有饮食，只有进食量
        '2027_0_20210521.csv', # 胰岛素基础量为中文
    
    ]
```

### 2.2.9 Pre-Processed Data

使用上述各类方法，我们对`Shanghai_T1DM`与`Shanghai_T2DM`两个数据集中的所有数据进行了预处理，结果保存在`processed-data`文件夹中。处理后数据的属性如下。

```
Date,CGM (mg / dl),Dietary intake,"CSII - bolus insulin (Novolin R, IU)","CSII - basal insulin (Novolin R, IU / H)",Insulin dose - s.c. insulin aspart 70/30,Insulin dose - s.c. insulin glarigine,Insulin dose - s.c. Gansulin R,Insulin dose - s.c. insulin aspart,Insulin dose - s.c. insulin aspart 50/50,Insulin dose - s.c. Humulin 70/30,Insulin dose - s.c. insulin glulisine,Insulin dose - s.c. Novolin 30R,Insulin dose - s.c. Novolin 50R,Insulin dose - s.c. insulin glargine,Insulin dose - s.c. Humulin R,Insulin dose - s.c. insulin degludec,Insulin dose - s.c. Gansulin 40R,Insulin dose - s.c. Novolin R,Insulin dose - s.c. insulin detemir,Non-insulin hypoglycemic agents acarbose,Non-insulin hypoglycemic agents gliquidone,Non-insulin hypoglycemic agents sitagliptin,Non-insulin hypoglycemic agents voglibose,Non-insulin hypoglycemic agents repaglinide,Non-insulin hypoglycemic agents liraglutide,Non-insulin hypoglycemic agents glimepiride,Non-insulin hypoglycemic agents pioglitazone,Non-insulin hypoglycemic agents canagliflozin,Non-insulin hypoglycemic agents dapagliflozin,Non-insulin hypoglycemic agents gliclazide,Non-insulin hypoglycemic agents metformin,Insulin dose - i.v. insulin aspart 70/30,Insulin dose - i.v. insulin glarigine,Insulin dose - i.v. Gansulin R,Insulin dose - i.v. insulin aspart,Insulin dose - i.v. insulin aspart 50/50,Insulin dose - i.v. Humulin 70/30,Insulin dose - i.v. insulin glulisine,Insulin dose - i.v. Novolin 30R,Insulin dose - i.v. Novolin 50R,Insulin dose - i.v. insulin glargine,Insulin dose - i.v. Humulin R,Insulin dose - i.v. insulin degludec,Insulin dose - i.v. Gansulin 40R,Insulin dose - i.v. Novolin R,Insulin dose - i.v. insulin detemir
```

# 3 血糖预测模型

