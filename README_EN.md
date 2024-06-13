# 2024_TJU_Data_Mining-Analysis_Report

# 1 Issue Analysis

## 1.1 Brief Introduction

This paper aims to construct a blood glucose level time series prediction model to assist diabetes patients in managing their blood sugar levels. Firstly, we delve into understanding the mechanisms of blood glucose regulation, factors influencing blood glucose levels, and the management methods for different types of diabetes. Secondly, based on the relevant data from the `Shanghai_T1DM` and `Shanghai_T2DM` datasets, we perform data cleaning, handle missing and outlier values, and conduct necessary data transformations and normalization. In terms of feature engineering, we select or construct features critical to glucose prediction based on domain knowledge, explore correlations among features, and perform feature selection or dimensionality reduction to enhance model efficiency. Next, we investigate and compare machine learning models suitable for time series glucose prediction, with the flexibility to choose methods from the latest papers or classical and effective approaches. The goal is to implement the selected model and fine-tune parameters to optimize prediction performance. Subsequently, we evaluate the model's predictive accuracy and generalization capacity using techniques such as cross-validation, AUC-ROC curves, and mean squared error. Finally, we conduct a thorough analysis of prediction outcomes, examining the impact of different factors on prediction accuracy, and present prediction results and model performance clearly using graphs and visualization tools. The aim is to predict blood glucose levels at 15, 30, 45, and 60 minutes and document the analysis results in this report.

## 1.2 Current Mainstream Blood Glucose Prediction Methods

## 1.3 Team Member

2151617 Zheng Zhi 2152970 Li Jinlin 2154306 Li Zekai 2154314 Zheng Kai

# 2 Dataset Analysis

## 2.1 Dataset Introduction

The datasets `ShanghaiT1DM` and `ShanghaiT2DM` comprise two folders named `Shanghai_T1DM` and `Shanghai_T2DM` and two summary sheets named `Shanghai_T1DM_Summary.csv` and `Shanghai_T2DM_Summary.csv`.

The `Shanghai_T1DM` folder and `Shanghai_T2DM` folder contain 3 to 14 days of `CGM` data corresponding to 12 patients with T1DM and 100 patients with T2DM, respectively. Of note, for one patient, there might be multiple periods of `CGM` recordings due to different visits to the hospital, which were stored in different excel tables. In fact, collecting data from different periods in one patient can reflect the changes of diabetes status during the follow-up. The excel table is named by the `patient ID`, `period number` and the `start date` of the `CGM` recording. Thus, for 12 patients with T1DM, there are 8 patients with 1 period of the `CGM` recording and 2 patients with 3 periods, totally equal to 16 excel tables in the `Shanghai_T1DM` folder. As for 100 patients with T2DM, there are 94 patients with 1 period of `CGM` recording, 6 patients with 2 periods, and 1 patient with 3 periods, amounting to 109 excel tables in the `Shanghai_T2DM` folder. Overall, the excel tables include `CGM` BG values every 15 minutes, capillary blood glucose `CBG` values, `blood ketone`, self-reported `dietary intake`, `insulin doses` and `non-insulin hypoglycemic agents`. The `blood ketone` was measured when diabetic ketoacidosis was suspected with a considerably high glucose level. Insulin administration includes continuous subcutaneous insulin infusion using insulin pump, multiple daily injections with insulin pen, and insulin that were given intravenously in case of an extremely high BG level.

Each excel table in the `Shanghai_T1DM` folder and `Shanghai_T2DM` folder contains the following data fields: `Date` Recording time of the `CGM` data. `CGM` CGM data recorded every 15 minutes. `CBG` CBG level measured by the glucose meter. `Blood ketone` Plasma-hydroxybutyrate measured with ketone test strips (Abbott Laboratories, Abbott Park, Illinois, USA). `Dietary intake` Self-reported time and weighed food intake `Insulin dose-s.c.` Subcutaneous insulin injection with insulin pen. `Insulin dose-i.v.` Dose of intravenous insulin infusion. `Non-insulin hypoglycemic agents` Hypoglycemic agents other than insulin. `CSII-bolus insulin` Dose of insulin delivered before a meal through insulin pump. `CSII-basal insulin `The rate (iu/per hour) at which basal insulin was continuously infused through insulin pump.

The summary sheets summarize the clinical characteristics, laboratory measurements and medications of the patients included in this study, with each row corresponding to one excel table in `Shanghai_T1DM` and `Shanghai_T2DM `folders. Clinical characteristics include `patient ID`, `gender`, `age`, `height`, `weight`, `BMI`, `smoking and drinking history`, `type of diabetes`, `duration of diabetes`, `diabetic complications`, `comorbidities` as well as `occurrence of hypoglycemia`. Laboratory measurements contain fasting and 2-hour postprandial plasma glucose/C-peptide/insulin, hemoglobin A1c (HbA1c), glycated albumin, total cholesterol, triglyceride, high-density lipoprotein cholesterol, low-density lipoprotein cholesterol, creatinine, estimated glomerular filtration rate, uric acid and blood urea nitrogen. Both hypoglycemic agents and medications given for other diseases before the `CGM` reading were also recorded.

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

## 3.1 Model Construction

To predict the blood glucose levels of any patient at 15, 30, 45, and 60 minutes based on the time series and static data provided in the `ShanghaiT1DM` and `ShanghaiT2DM` datasets, we chose the `TCN` layer for processing all time series data due to its outstanding performance in capturing dependencies and features in sequential data through convolution and residual connections. This includes `drug concentrations at any given time`, `medication information`, `dietary intake`, and the patient's previous `CGM` data. Additionally, the `Flatten` layer ensures that the embedding vectors of the time series data are flattened into structured N-dimensional vectors required for subsequent processing.

```python
ts_input = Input(shape=(ts_X_train.shape[1], ts_X_train.shape[2]))
ts_embedding = TCN(64)(ts_input)
ts_embedding = Flatten()(ts_embedding)
```

All static data, including the patient's `ID`, `height`, `weight`, and `gender`, is processed through a simple `Dense` layer. This ensures that the static data's influence on blood glucose levels is considered by the prediction model and has the same embedding dimension N as the processed time series data.

```python
static_input = Input(shape=(static_X_train.shape[1],))
static_embedding = Dense(64, activation='relu')(static_input)
```

Next, we designed a `cross_attention` layer that calculates the attention scores between the `ts_embedding` of the time series data and the `static_embedding` of the static data using matrix multiplication. The scores are then processed with `Softmax` to obtain weights, which are used to compute a weighted sum. After flattening, the output is a structured N-dimensional vector. In this way, the N-dimensional time series data and static data are encoded. By utilizing the `TCN`, `Dense`, and `cross_attention` layers, the model comprehensively considers the influence of all factors on blood glucose levels.

```python
def cross_attention(x1, x2):
    attention_scores = tf.matmul(x1, x2, transpose_b=True)
    attention_weights = tf.nn.softmax(attention_scores, axis=-1)
    attended_vector = tf.matmul(attention_weights, x2)
    return attended_vector

cross_attention_output = cross_attention(tf.expand_dims(ts_embedding, axis=1), tf.expand_dims(static_embedding, axis=1))
cross_attention_output = Flatten()(cross_attention_output)
```

In the decoding part, we further process the previously encoded N-dimensional vector using a `Dense` layer with an output dimension of 64 and a ReLU activation function. Then, we output four target values to predict the blood glucose levels for the next 15, 30, 45, and 60 minutes.

```python
output = Dense(64, activation='relu')(cross_attention_output)
output = Dense(4)(output)  # Output Layer
```

Finally, we use the functions provided in the `TensorFlow` library to build, compile, and run the aforementioned model.

```python
model = Model(inputs=[ts_input, static_input], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
model.summary()
```

## 3.2 Model Deployment

Due to the computational limitations of the local machine and to save time, we chose to deploy the model on the cloud-based `AutoDL` platform. We rented an `RTX 4090D (24GB)` GPU, with a CPU configuration of `15 vCPU Intel(R) Xeon(R) Platinum 8474C`, on an hourly basis. The image version used was `TensorFlow 2.9.0 Python 3.8 (ubuntu 20.04) Cuda 11.2`. The specific configuration of the cloud model container is shown in the following image:

![pic1](/pic/1.png)

We then used `JupyterLab` to upload the model code constructed in section 3.1 and the preprocessed data from section 2.2.10 to the cloud container instance.


