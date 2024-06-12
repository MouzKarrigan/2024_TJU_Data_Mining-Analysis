# 2024_TJU_Data_Mining-Analysis_Report

# 1 Issue Analysis

## 1.1 Brief Introduction

## 1.2 Current Mainstream Blood Glucose Prediction Methods

## 1.3 

## 1.4 Team Member

2151617 Zheng Zhi 2152970 Li Jinlin 2154306 Li Zekai 2154314 Zheng Kai

# 2 Dataset Analysis

## 2.1 Dataset Introduction

## 2.2 Data Pre-process

### 2.2.1 Format Transformation

For ease of subsequent pre-processing, first convert all .xlsx files in the T1DM and T2DM datasets to .csv files and store them.

```python
for file in T1DM:
    if file == '.DS_Store':
        continue
    data = pd.read_excel(os.path.join(T1DM_folder_url, file))

    new_filename = file.split('.')[0] + '.csv'
    data.to_csv(os.path.join(new_T1DM_folder_url, new_filename))
```

### 2.2.2 Dietary Intake Pre-process

In the original dataset, there are two attributes representing the patient's `Dietary Intake` content and its Chinese translation. We will combine these two attributes into one, simply determining whether there is content in the original `Dietary Intake` data. If there is content, it indicates that the patient is eating at that time, and we mark it as `1` in the output data's `Dietary Intake`. If there is no content, we mark it as `0` in the output data.

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

We ignore the specific content of the meals and only consider whether the patient is eating or not, aiming to simplify the subsequent model construction and computation. Given the limited data, analyzing the average impact of eating on blood glucose levels is more accurate and convenient than analyzing the varying effects of each specific nutrient intake on blood glucose levels.

### 2.2.3 Agent Details Pre-process

In the original dataset, `Shanghai_T1DM_Summary.csv` and `Shanghai_T2DM_Summary.csv` provide information on all the medications used in the dataset and whether they contain insulin. To facilitate subsequent data preprocessing, we identified and extracted all medication names using `agents_process.py` and saved them in `agents_info.json`.

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
Then we extracted and saved the names of insulin-containing antidiabetic medications to `insulin_agents.json`, and the names of non-insulin antidiabetic medications to `non_insulin_agents.json`.

### 2.2.4 Insulin Dose - s.c. Pre-process

In the original dataset, there is an attribute `Insulin Dose - s.c.` that records the medication name and dosage administered via subcutaneous injection in the format `Novolin R, 2 IU`, with the medication name before the comma and the dosage after the comma. To handle this, we designed a method to separately extract the medication name and dosage from the `Insulin Dose - s.c.` attribute and record them in the output data.

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

It is worth noting that a single injection may not always involve only one type of medication but could simultaneously include two or more different types. Therefore, in the preprocessed output data, we do not use the conventional method of converting the `Insulin Dose - s.c.` attribute into `Insulin Kind` and `Dose` attributes. Instead, we set all the insulin-containing antidiabetic medication names that appear in `insulin_agents.json` as new attributes. In the output data, we only record the dosage under the corresponding medication type attribute, with the unit in IU. If a particular medication is not present, the dosage is recorded as `0`. This approach allows us to handle cases where two or more different types of medications are injected simultaneously.

### 2.2.5 Insulin Dose - i.v. Pre-process

In the original dataset, the attribute `Insulin Dose - s.c.` records the names and doses of medications administered to patients via intravenous injection, similar to `500ml 0.9% sodium chloride, 12 IU Novolin R, 10 ml 10% potassium chloride`. We disregard the irrelevant auxiliary medication components before and after, extracting only the main components of antidiabetic medications that affect blood glucose concentration from the middle of each record.

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

The processing procedure is similar to that of 2.2.4. All antidiabetic medication names appearing in `insulin_agents.json` are set as new attributes separately. In the output data, only the doses are recorded under the corresponding medication type attribute, with the unit being IU. If there is no such medication, the dose is recorded as `0`. It's worth noting that, to differentiate between intravenous and subcutaneous injections, the newly added medication dose attributes will have prefixes `Insulin dose - s.c.` or `Insulin dose - i.v.` based on the injection form, facilitating subsequent model calculations.

### 2.2.6 Non-insulin Hypoglycemic Agents Pre-process

In the original dataset, the attribute `Non-insulin hypoglycemic agents` records the types and doses of non-insulin hypoglycemic agents consumed by patients. Our preprocessing of this attribute is similar to that of sections 2.2.4 and 2.2.5.

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

All newly added attributes in the output data are formed by prefixing the drug name with `Non-insulin hypoglycemic agents`, distinguishing them from the new attributes added in the processes of sections 2.2.4 and 2.2.5. These attributes only record the dosage of the respective drug.

### 2.2.7 CSII Pre-process

In the original dataset, the attribute `CSII - bolus insulin (Novolin R, IU)` and `CSII - basal insulin (Novolin R, IU / H)` exist. The latter represents the continuous basal insulin dose within a period, presented in merged cells in the .xlsx file. When converted to .csv format, some data loss occurred. We devised a method to address this issue by completing the missing data. Similarly, in the output data, the entire period's `CSII - basal insulin (Novolin R, IU / H)` is recorded, but each entry is recorded separately instead of being merged.

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

It's worth noting that in both `CSII - bolus insulin (Novolin R, IU)` and `CSII - basal insulin (Novolin R, IU / H)` attributes, there might be records indicating "temporarily suspend insulin delivery," signifying a pause in continuous medication delivery for the next short period. We devised a method to identify and record the subsequent period's dose as `0` in the output data until new dosage information is available.

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

In the original dataset, other attributes such as `Date` and `CGM (mg/dl)` are directly usable, so we transferred them to the output data.

```python
def read_csv(url) -> DataFrame:
    return pd.read_csv(url)

def save_csv(df: DataFrame, url: str):
    df.to_csv(url)
```

But the attribute `CBG (mg/dl)` refers to blood glucose concentration measured in another way, with many empty values that are difficult to handle in the model. Also, it overlaps with `CGM (mg/dl)`, so we removed the `CBG (mg/dl)` attribute from the output data. `Blood Ketone (mmol/L)` indicates blood ketone levels, which have poor relevance to the problem at hand and many missing values, making it challenging to handle. Therefore, we also removed this attribute from the output data.

```python
def process_CBG_blood_ketone(df:DataFrame) -> DataFrame:
    return df.drop(columns=['CBG (mg / dl)', 'Blood Ketone (mmol / L)'])
```

### 2.2.9 Data Select

In several .xlsx files in the original dataset, some tables contain serious formatting errors due to oversight by the researchers during recording. Considering the minimal impact on a very small portion of the data and the associated development costs, we opted not to devise additional methods to handle data with severe formatting issues but to exclude them directly. The specific actions and reasons are as follows.

```python
ban_url = [
    '2045_0_20201216.csv',  # CSII - bolus insulin without unit
    '2095_0_20201116.csv',  # CSII - bolus insulin without unit
    '2013_0_20220123.csv',  # No dietary intake, only meal intake
    '2027_0_20210521.csv',  # Basal insulin dose in Chinese

]
```

### 2.2.10 Pre-Processed Data

Using the various methods mentioned above, we preprocessed all the data in the `Shanghai_T1DM` and `Shanghai_T2DM` datasets, and the results are saved in the `processed-data` folder. The attributes of the processed data are as follows.

```
Date,CGM (mg / dl),Dietary intake,"CSII - bolus insulin (Novolin R, IU)","CSII - basal insulin (Novolin R, IU / H)",Insulin dose - s.c. insulin aspart 70/30,Insulin dose - s.c. insulin glarigine,Insulin dose - s.c. Gansulin R,Insulin dose - s.c. insulin aspart,Insulin dose - s.c. insulin aspart 50/50,Insulin dose - s.c. Humulin 70/30,Insulin dose - s.c. insulin glulisine,Insulin dose - s.c. Novolin 30R,Insulin dose - s.c. Novolin 50R,Insulin dose - s.c. insulin glargine,Insulin dose - s.c. Humulin R,Insulin dose - s.c. insulin degludec,Insulin dose - s.c. Gansulin 40R,Insulin dose - s.c. Novolin R,Insulin dose - s.c. insulin detemir,Non-insulin hypoglycemic agents acarbose,Non-insulin hypoglycemic agents gliquidone,Non-insulin hypoglycemic agents sitagliptin,Non-insulin hypoglycemic agents voglibose,Non-insulin hypoglycemic agents repaglinide,Non-insulin hypoglycemic agents liraglutide,Non-insulin hypoglycemic agents glimepiride,Non-insulin hypoglycemic agents pioglitazone,Non-insulin hypoglycemic agents canagliflozin,Non-insulin hypoglycemic agents dapagliflozin,Non-insulin hypoglycemic agents gliclazide,Non-insulin hypoglycemic agents metformin,Insulin dose - i.v. insulin aspart 70/30,Insulin dose - i.v. insulin glarigine,Insulin dose - i.v. Gansulin R,Insulin dose - i.v. insulin aspart,Insulin dose - i.v. insulin aspart 50/50,Insulin dose - i.v. Humulin 70/30,Insulin dose - i.v. insulin glulisine,Insulin dose - i.v. Novolin 30R,Insulin dose - i.v. Novolin 50R,Insulin dose - i.v. insulin glargine,Insulin dose - i.v. Humulin R,Insulin dose - i.v. insulin degludec,Insulin dose - i.v. Gansulin 40R,Insulin dose - i.v. Novolin R,Insulin dose - i.v. insulin detemir
```

# 3 Blood Glucose Prediction Model

