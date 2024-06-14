# 2024_TJU_Data_Mining-Analysis_Report

# 1 问题分析

## 1.1 问题概述

本文旨在构建血糖水平时间序列预测模型，以帮助糖尿病患者管理血糖。首先，我们深入了解血糖调节机制、影响血糖水平的因素、以及不同类型糖尿病的管理方法。其次，我们根据`Shanghai_T1DM`和`Shanghai_T2DM`两个数据集中的相关数据，进行数据清洗、处理缺失值和异常值，进行必要的数据转换和归一化。在特征工程方面，我们根据领域知识选择或构建对血糖预测至关重要的特征，并探索特征之间的相关性，进行特征选择或降维以提高模型效率。然后，我们研究和比较适用于时间序列血糖预测的机器学习模型，选择合适的模型可以基于最新论文中的方法或经典有效的方法。模型的选择方向不限，目的是实现所选模型并调整参数以优化预测性能。接下来，评估模型的预测准确性和泛化能力，使用交叉验证、AUC-ROC曲线和均方误差等技术。最后，我们进行深入的预测结果分析，考察不同因素对血糖预测准确性的影响，并同时使用图表和可视化工具清晰地呈现预测结果和模型性能，预测15、30、45和60分钟后的血糖水平，并在本文中写下分析结果。

## 1.2 当前主流血糖预测方法

### 1.2.1 基于非线性自回归（NAR）神经网络

根据`Alessandro Aliberti`及其团队的[研究成果](https://ieeexplore.ieee.org/abstract/document/8723121)，他们基于非线性自回归（NAR）神经网络开发了一种高效的血糖预测方法。相比传统的线性自回归模型，`NAR`模型不受分布限制，能够处理血糖数据中的非线性特征，如突发的波动和短暂的瞬态期。该方法通过将过去的血糖值作为输入，使用多层神经网络计算未来时刻的血糖值。初步设计包括一个输入层、一个隐藏层和一个输出层，通过`Levenberg-Marquardt`反向传播过程进行学习。为了优化模型，他们采用了Lipschitz方法确定最佳回归器数量，并通过`Optimal Brain Surgeon`方法进行自动修剪，以提高模型的紧凑性和泛化能力。经过实验验证，他们选择了具有8个回归器的最终`NAR`模型，以实现精确且高效的血糖水平预测。

### 1.2.2 基于长短期记忆（LSTM）网络与递归神经网络（RNN）

根据`Mario Munoz-Organero`及其团队的[研究成果](https://www.mdpi.com/1424-8220/20/14/3896)，他们基于长短期记忆（LSTM）和递归神经网络（RNN）模型，预测血糖水平的变化。模型综合考虑当前血糖水平、碳水化合物摄入和胰岛素注射的影响，使用微分方程描述血糖代谢过程，包括消化、吸收、胰岛素依赖和独立利用、肾脏清除和内源性肝脏产生。通过训练递归神经网络（RNN）学习碳水化合物消化和胰岛素吸收过程，再通过`LSTM`模型学习血糖水平的时间模式，结合处理后的胰岛素和碳水化合物信号，估计未来连续血糖监测（CGM）读数的血糖变化。模型在不考虑外部输入时，基于代谢动态生成预测，提前警示可能出现的负面情况，并推荐避免这些情况的具体措施。

### 1.2.3 基于自回归移动平均（ARMA）模型

根据`Ning Ma`及其团队的[研究成果](https://ceur-ws.org/Vol-2675/paper27.pdf),他们的ARMA模型结合了自回归（AR）和移动平均（MA）部分，用于捕捉时间序列数据的线性和非线性动态。模型首先通过滑动窗口技术更新输入数据，并利用`AIC`和`BIC`确定最优模型阶数。随后，预测结果与原始数据进行比较，得到残差时间序列。接着，利用BP神经网络对残差进行预测，以提高模型对非线性成分的适应能力。最终，将`ARMA`模型和残差补偿网络的预测结果结合，通过统计分析得出最终的血糖预测值。这一框架能有效处理血糖数据中的线性和非线性特征，提高了预测精度和稳定性，适用于实时血糖控制和管理。

## 1.3 小组人员构成

2151617 郑埴 2152970 李锦霖 2154306 李泽凯 2154314 郑楷

## 1.4 代码管理

我们采用`Github`来进行本项目的代码管理与版本控制，所有的源代码、环境、报告文档与PPT都存储在以下仓库中：

[github.com/MouzKarrigan/2024_TJU_Data_Mining-Analysis](https://github.com/MouzKarrigan/2024_TJU_Data_Mining-Analysis)

# 2 数据集分析

## 2.1 所用数据集介绍

数据集`ShanghaiT1DM`和`ShanghaiT2DM`包含两个文件夹，分别命名为`Shanghai_T1DM`和`Shanghai_T2DM`，以及两个汇总表，分别命名为`Shanghai_T1DM_Summary.csv`和`Shanghai_T2DM_Summary.csv`。

`Shanghai_T1DM`文件夹和`Shanghai_T2DM`文件夹包含了与12名T1DM患者和100名T2DM患者对应的3至14天的`CGM`数据。值得注意的是，对于一个患者，可能会由于多次就医而有多个`CGM`记录的时期，这些记录存储在不同的excel表格中。实际上，从一个患者的不同时期收集数据可以反映出随访期间糖尿病状态的变化。excel表格以患者ID、时期编号和CGM记录的开始日期命名。因此，对于12名T1DM患者，有8名患者有1个CGM记录时期，有2名患者有3个时期，总共有16个excel表格在`Shanghai_T1DM`文件夹中。至于100名T2DM患者，有94名患者有1个CGM记录时期，有6名患者有2个时期，有1名患者有3个时期，总共有109个excel表格在`Shanghai_T2DM`文件夹中。总体而言，excel表格包括每15分钟的CGM血糖值、毛细血管血糖CBG值、血酮、自报饮食摄入、胰岛素剂量和非胰岛素降糖药物。当怀疑发生糖尿病酮症酸中毒时，会测量血酮，其血糖水平相当高。胰岛素给药包括使用胰岛素泵进行连续皮下注射、使用胰岛素笔进行多次每日注射，以及在血糖水平极高的情况下静脉注射胰岛素。

`Shanghai_T1DM`文件夹和`Shanghai_T2DM`文件夹中的每个excel表格包含以下数据字段：`Date` CGM数据的记录时间。`CGM` 每15分钟记录的CGM数据。`CBG` 血糖仪测量的CBG水平。`Blood ketone` 使用酮体试纸（Abbott Laboratories，Abbott Park，Illinois，USA）测量的血浆羟丁酸。`Dietary intake` 自报的进食时间和称重的食物摄入量。`Insulin dose-s.c.` 使用胰岛素笔进行的皮下注射。`Insulin dose-i.v.` 静脉注射胰岛素的剂量。`Non-insulin hypoglycemic agents` 除胰岛素外的降糖药物。`CSII-bolus insulin` 进餐前通过胰岛素泵输送的胰岛素剂量。`CSII-basal insulin` 通过胰岛素泵持续输送的基础胰岛素的速率（iu/每小时）。

汇总表总结了本研究中所包括的患者的临床特征、实验室检查和药物治疗，每一行对应于`Shanghai_T1DM`和`Shanghai_T2DM`文件夹中的一个excel表格。临床特征包括患者ID、性别、年龄、身高、体重、BMI、吸烟和饮酒史、糖尿病类型、糖尿病持续时间、糖尿病并发症、合并症，以及低血糖的发生情况。实验室检查包括空腹和餐后2小时的血浆葡萄糖/C肽/胰岛素、糖化血红蛋白（HbA1c）、糖化白蛋白、总胆固醇、甘油三酯、高密度脂蛋白胆固醇、低密度脂蛋白胆固醇、肌酐、估算的肾小球滤过率、尿酸和血尿素氮。在`CGM`读数之前，也记录了用于其他疾病的降糖药物和药物。

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

### 2.2.9 Data Select

在原数据集中的若干.xlsx里，有几份表格因为研究人员的疏忽在记录时出现了严重的格式错误，由于只有极少部分的数据受到影响，因此本着开发成本上的考量，我们不再额外设计方法处理这类格式出现严重问题的数据，而是直接将它们排除。具体理由如下。

```python
ban_url = [
        '2045_0_20201216.csv', # CSII -bolus insulin没有单位
        '2095_0_20201116.csv' ,# CSII -bolus insulin没有单位
        '2013_0_20220123.csv', # 没有饮食，只有进食量
        '2027_0_20210521.csv', # 胰岛素基础量为中文
    
    ]
```

### 2.2.10 Pre-Processed Data

使用上述各类方法，我们对`Shanghai_T1DM`与`Shanghai_T2DM`两个数据集中的所有数据进行了预处理，结果保存在`processed-data`文件夹中。处理后数据的属性如下。

```
Date,CGM (mg / dl),Dietary intake,"CSII - bolus insulin (Novolin R, IU)","CSII - basal insulin (Novolin R, IU / H)",Insulin dose - s.c. insulin aspart 70/30,Insulin dose - s.c. insulin glarigine,Insulin dose - s.c. Gansulin R,Insulin dose - s.c. insulin aspart,Insulin dose - s.c. insulin aspart 50/50,Insulin dose - s.c. Humulin 70/30,Insulin dose - s.c. insulin glulisine,Insulin dose - s.c. Novolin 30R,Insulin dose - s.c. Novolin 50R,Insulin dose - s.c. insulin glargine,Insulin dose - s.c. Humulin R,Insulin dose - s.c. insulin degludec,Insulin dose - s.c. Gansulin 40R,Insulin dose - s.c. Novolin R,Insulin dose - s.c. insulin detemir,Non-insulin hypoglycemic agents acarbose,Non-insulin hypoglycemic agents gliquidone,Non-insulin hypoglycemic agents sitagliptin,Non-insulin hypoglycemic agents voglibose,Non-insulin hypoglycemic agents repaglinide,Non-insulin hypoglycemic agents liraglutide,Non-insulin hypoglycemic agents glimepiride,Non-insulin hypoglycemic agents pioglitazone,Non-insulin hypoglycemic agents canagliflozin,Non-insulin hypoglycemic agents dapagliflozin,Non-insulin hypoglycemic agents gliclazide,Non-insulin hypoglycemic agents metformin,Insulin dose - i.v. insulin aspart 70/30,Insulin dose - i.v. insulin glarigine,Insulin dose - i.v. Gansulin R,Insulin dose - i.v. insulin aspart,Insulin dose - i.v. insulin aspart 50/50,Insulin dose - i.v. Humulin 70/30,Insulin dose - i.v. insulin glulisine,Insulin dose - i.v. Novolin 30R,Insulin dose - i.v. Novolin 50R,Insulin dose - i.v. insulin glargine,Insulin dose - i.v. Humulin R,Insulin dose - i.v. insulin degludec,Insulin dose - i.v. Gansulin 40R,Insulin dose - i.v. Novolin R,Insulin dose - i.v. insulin detemir
```

# 3 血糖预测模型

## 3.1 模型构建

为了依据`ShanghaiT1DM`和`ShanghaiT2DM`数据集中所给的的时序数据和静态数据，预测任意患者在15，30，45与60分钟后的血糖水平，而`LSTM`层在时序数据处理中表现出色，通过卷积和残差连接能够有效捕捉时序数据的依赖关系和特征，所以我们选择TCN层处理所有的时序数据，包括`各种药物在任意时间的浓度`、`给药信息`、`进食信息`以及患者的前序`CGM`。同时`Flatten`层确保时序数据的嵌入向量的像素被扁平化处理，成为后续所需的规整的N维向量。

```python
x = LSTM(64, return_sequences=True)(ts_input)
x = LSTM(56, return_sequences=True)(x)
x = LSTM(48, return_sequences=True)(x)
x = LSTM(40, return_sequences=True)(x)
x = LSTM(36, return_sequences=True)(x)
x = LSTM(32)(x)
ts_embedding = Flatten()(x)
```

而所有的静态数据，包括患者的`编号`、`身高`、`体重`和`性别`等，我们通过简单的`Dense`层对其进行处理，保证这类静态数据对血糖水平的影响也能被预测模型所考虑到的同时，与上述处理后的时序数据具有相同的嵌入维度N。

```python
y = Dense(64, activation='relu')(static_input)
y = Dense(56, activation='relu')(y)
y = Dense(48, activation='relu')(y)
y = Dense(40, activation='relu')(y)
y = Dense(36, activation='relu')(y)
y = Dense(32, activation='relu')(y)
y = Dense(32, activation='relu')(y)  # 保持 32 维
y = Dense(32, activation='relu')(y)  # 保持 32 维
static_embedding = y
```

之后我们通过设计一个`cross_attention`层，用矩阵乘法计算上述所得的时序数据`ts_embedding`和静态数据`static_embedding`之间的注意力得分，而后通过`Softmax`处理得到权重，加权求和并展平后输出规整的N维向量。这样便对上述的N维时序数据和静态数据进行了编码，利用`TCN`、`Dense`和`cross_attention`三个编码层融合考量了所有因素对血糖水平的影响。

```python
# Cross-Attention层
def cross_attention(query, key, value):
    attention_scores = tf.matmul(query, key, transpose_b=True)
    attention_weights = tf.nn.softmax(attention_scores, axis=-1)
    attended_vector = tf.matmul(attention_weights, value)
    return attended_vector

# 使用静态特征作为查询，时序特征作为键和值
query1 = tf.expand_dims(static_embedding, axis=1)
key1 = tf.expand_dims(ts_embedding, axis=1)
value1 = tf.expand_dims(ts_embedding, axis=1)
cross_attention_output1 = cross_attention(query1, key1, value1)
cross_attention_output1 = Flatten()(cross_attention_output1)

# 使用时序特征作为查询，静态特征作为键和值
query2 = tf.expand_dims(ts_embedding, axis=1)
key2 = tf.expand_dims(static_embedding, axis=1)
value2 = tf.expand_dims(static_embedding, axis=1)
cross_attention_output2 = cross_attention(query2, key2, value2)
cross_attention_output2 = Flatten()(cross_attention_output2)
```

在解码部分我们通过`Dense`层对前序编码的N维向量进行进一步处理，输出维度为64，激活函数为ReLU，然后输出4个目标值，预测未来15、30、45和60分钟的血糖水平。

```python
# 合并两个 cross-attention 输出
merged_attention_output = Concatenate()([cross_attention_output1, cross_attention_output2])

# 解码层，从合并后的维度逐渐减少到 4
z = Dense(64, activation='relu')(merged_attention_output)
z = Dense(56, activation='relu')(z)
z = Dense(48, activation='relu')(z)
z = Dense(40, activation='relu')(z)
z = Dense(36, activation='relu')(z)
z = Dense(32, activation='relu')(z)
z = Dense(16, activation='relu')(z)
output = Dense(4)(z)  # 输出层，预测4个目标值
```

最后通过`Tenserflow`库中封装函数构建、编译与运行上述的模型。

```python
model = Model(inputs=[ts_input, static_input], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
model.summary()
```

模型运行时将会读取前序步骤中预处理的数据，同时将对数据进行进一步的处理，分离数据中的时序特征和预测目标值，并将输入数据进行标准化操作方便后续训练。

```python
# 分离特征和目标值
time_series_features = data[time_serise_attribute].values
static_features = data[static_attribute].values
targets = data[target_attribute].values

def create_sequences(features, targets, static_features, time_steps=10):
    ts_X,  y = [], []
    for i in range(len(features) - time_steps):
        ts_X.append(features[i:i+time_steps])
        static_X = static_features[i]
        y.append(targets[i+time_steps])
    return np.array(ts_X), np.array(static_X), np.array(y)

ts_X, static_X, y = create_sequences(time_series_features, targets, static_features)

# 数据标准化
scaler_ts_X = StandardScaler()
ts_X = scaler_ts_X.fit_transform(ts_X.reshape(-1, ts_X.shape[-1])).reshape(ts_X.shape)

scaler_static_X = StandardScaler()
static_X = scaler_static_X.fit_transform(static_X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

self.ts_X_train, self.ts_X_test, self.static_X_train, self.static_X_test, self.y_train, self.y_test = train_test_split(
    ts_X, static_X, y, test_size=0.2, random_state=42
)
```

## 3.2 模型部署

由于本地机器的算力限制，为了节约时间成本，我们选择将模型部署在云端的`AutoDL`平台上。我们按时租用了一张`RTX 4090D(24GB)`的GPU，CPU配置为`15 vCPU Intel(R) Xeon(R) Platinum 8474C`，镜像版本为`TensorFlow  2.9.0 Python  3.8(ubuntu20.04) Cuda  11.2`。云端模型容器的具体配置如下图所示：

![pic1](/pic/1.png)

之后我们采用`JupyterLab`将3.1中的已构建的模型代码与2.2.10中预处理产出的数据上传至云端容器实例中。下一步，在云端服务器中安装模型所需的`sklearn`库和`pandas`库，由于服务器配置时自带`Tensorflow`库因此不用安装。

![pic2](/pic/2.png)

![pic3](/pic/3.png)

## 3.3 模型训练

我们在所有经过预处理和格式标准化的数据中，通过`sklearn`库中自带的`train_test_split`划分了大比例的训练集，并用这些训练集对上述模型进行训练。

```python
from sklearn.model_selection import train_test_split

self.ts_X_train, self.ts_X_test, self.static_X_train, self.static_X_test, self.y_train, self.y_test = train_test_split(
            ts_X, static_X, y, test_size=0.2, random_state=42)
```

在经过5-7次的微调与每次微调后在云端服务的二十分钟左右训练后，我们得到了可以有效根据时序数据预测后续血糖浓度水平的模型，将其命名为`GCM_model.h5`后导出并保存至本地，以便后续的预测以及能力评估。

# 4 模型预测能力评估

## 4.1 基于均值绝对误差（MAE）评估

### 4.1.1 评估模型介绍

均值绝对误差（MAE）是一种常用的衡量预测模型误差的方法。`MAE`计算预测值与实际值之间的绝对差异的平均值。`MAE`的优点在于它简单易懂，且对异常值不敏感。由于`MAE`只是简单地计算绝对误差的平均值，因此每个误差的贡献是线性的，这使得`MAE`更能反映实际的预测误差水平。`MAE`值越小，表明模型的预测能力越强。

### 4.1.2 评估过程

我们仍然通过`sklearn`库中自带的`train_test_split`划分了小比例的测试集，并将测试集投入已经训练好的血糖浓度预测模型`GCM_model.h5`中，并设计了方法，利用`MAE`模型评估`GCM_model.h5`根据测试集做出的血糖预测值与测试集的血糖实际值之间的误差水平。

```python
def evaluate_model(self):
        test_loss, test_mae = self.model.evaluate([self.ts_X_test, self.static_X_test], self.y_test)
        print(f'Test loss: {test_loss}, Test MAE: {test_mae}')
        return f'Test loss: {test_loss}, Test MAE: {test_mae}'
```

我们将每个`epoch`预测值相比于真实值的`loss`和`mae`记录在`model_info.log`中并导出保存，方便后续直观评估模型的预测能力并进行可视化；同时直接在控制台中输出前`5`个预测值与前`5`个真实值，方便及时发现模型的严重错误并决定是否继续进行调优。

```python
with open("model_info.log", 'w') as file:
        # 打印每个epoch的loss和mae
        for epoch, (loss, mae) in enumerate(zip(history.history['loss'], history.history['mean_absolute_error'])):
            print(f"Epoch {epoch + 1}: Loss = {loss}, MAE = {mae}")
            file.write(f"Epoch {epoch + 1}: Loss = {loss}, MAE = {mae}\n")
        file.write("Evaluate Result: " + evaluate_result + '\n')

# 打印预测效果
print(f'Predictions: {y_pred[:5]}')  # 仅显示前5个预测值
print(f'Actual: {y_test[:5]}')  # 仅显示前5个真实值
```

### 4.1.3 评估结果&可视化

保存在`model_info.log`中的对本模型的基于均值绝对误差（MAE）的评估结果如下：

```log
Epoch 1: Loss = 0.24813230335712433, MAE = 0.3392144441604614
Epoch 2: Loss = 0.17603440582752228, MAE = 0.28965243697166443
Epoch 3: Loss = 0.16945107281208038, MAE = 0.28416118025779724
Epoch 4: Loss = 0.16450119018554688, MAE = 0.2798997163772583
Epoch 5: Loss = 0.1617734283208847, MAE = 0.2778536379337311
Epoch 6: Loss = 0.15967202186584473, MAE = 0.2754196226596832
Epoch 7: Loss = 0.15808287262916565, MAE = 0.27424079179763794
Epoch 8: Loss = 0.15543019771575928, MAE = 0.2722415626049042
Epoch 9: Loss = 0.152229905128479, MAE = 0.2698534429073334
Epoch 10: Loss = 0.15068094432353973, MAE = 0.2676985263824463
Epoch 11: Loss = 0.1485109180212021, MAE = 0.2659306228160858
Epoch 12: Loss = 0.14616045355796814, MAE = 0.264129638671875
Epoch 13: Loss = 0.14481528103351593, MAE = 0.2629924714565277
Epoch 14: Loss = 0.1432819664478302, MAE = 0.2612338960170746
Epoch 15: Loss = 0.14213888347148895, MAE = 0.26077672839164734
Epoch 16: Loss = 0.1405232697725296, MAE = 0.2589830756187439
Epoch 17: Loss = 0.13920897245407104, MAE = 0.2579362392425537
Epoch 18: Loss = 0.13700328767299652, MAE = 0.2558068633079529
Epoch 19: Loss = 0.135841503739357, MAE = 0.25503242015838623
Epoch 20: Loss = 0.1343383491039276, MAE = 0.25366151332855225
Epoch 21: Loss = 0.13319242000579834, MAE = 0.25238853693008423
Epoch 22: Loss = 0.13157425820827484, MAE = 0.25120872259140015
Epoch 23: Loss = 0.13002479076385498, MAE = 0.24965637922286987
Epoch 24: Loss = 0.12899799644947052, MAE = 0.2487075924873352
Epoch 25: Loss = 0.12715759873390198, MAE = 0.24738220870494843
Epoch 26: Loss = 0.1254996806383133, MAE = 0.24577780067920685
Epoch 27: Loss = 0.12392573058605194, MAE = 0.24451224505901337
Epoch 28: Loss = 0.12294596433639526, MAE = 0.2437702715396881
Epoch 29: Loss = 0.12211327999830246, MAE = 0.24265168607234955
Epoch 30: Loss = 0.11990866810083389, MAE = 0.2406342476606369
Epoch 31: Loss = 0.11868849396705627, MAE = 0.23954859375953674
Epoch 32: Loss = 0.11719253659248352, MAE = 0.2382337599992752
Epoch 33: Loss = 0.11625178158283234, MAE = 0.23720043897628784
Epoch 34: Loss = 0.11457919329404831, MAE = 0.23559845983982086
Epoch 35: Loss = 0.11278732120990753, MAE = 0.23400256037712097
Epoch 36: Loss = 0.111522376537323, MAE = 0.23233428597450256
Epoch 37: Loss = 0.11087394505739212, MAE = 0.23186525702476501
Epoch 38: Loss = 0.10904955863952637, MAE = 0.23011435568332672
Epoch 39: Loss = 0.10804162174463272, MAE = 0.22891177237033844
Epoch 40: Loss = 0.1072082668542862, MAE = 0.22841022908687592
Epoch 41: Loss = 0.10584983229637146, MAE = 0.22709250450134277
Epoch 42: Loss = 0.1048450917005539, MAE = 0.22595353424549103
Epoch 43: Loss = 0.1035393550992012, MAE = 0.2244720607995987
Epoch 44: Loss = 0.10208194702863693, MAE = 0.22328290343284607
Epoch 45: Loss = 0.10101437568664551, MAE = 0.22216375172138214
Epoch 46: Loss = 0.09999130666255951, MAE = 0.22099186480045319
Epoch 47: Loss = 0.09851470589637756, MAE = 0.21978960931301117
Epoch 48: Loss = 0.09900260716676712, MAE = 0.22022844851016998
Epoch 49: Loss = 0.09660875797271729, MAE = 0.21708492934703827
Epoch 50: Loss = 0.0960533395409584, MAE = 0.2172061800956726
Evaluate Result: Test loss: 0.143349751830101, Test MAE: 0.2549135386943817
```

可见评估后的`loss`与`MAE`均处在相对较低的水平，说明`GCM_model.h5`模型的血糖预测能力较强，达到了项目的预期目标。

为了更加直观显示`GCM_model.h5`模型的血糖预测能力以及其`MAE`的波动范围，我们在`MAE_curve.py`中，利用`matplotlib.pyplot`库对其进行曲线可视化：

```python
plt.figure(figsize=(10, 6))
plt.plot(epochs, mae, marker='o', linestyle='-')
plt.title('MAE Curve')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.grid(True)
plt.show()
```

评估结果的可视化如下：

![MAE curve](./pic/MAE_Curve.png)

由于该曲线是训练过程中生成的，而训练过程中的数据是经过3.3的格式标准化后的，难免产生误差，我们最终利用原始数据重新计算了`MAE`水平，具体实现过程请看4.3.3，结果如下：

![MAE_MSE result](./pic/MAE_MSE_Result.png)

## 4.2 基于全数据集的交叉验证评估

### 4.2.1 评估过程

为了减少极端数据带来的影响，增大测试的数据基数同时更方便生成全览的可视化结果，我们在`model_test.py`中将整个数据集都作为测试集投入之前训练好的`GCM_model.h5`模型中，并将模型给出的预测数据与整个数据集的真实值进行对比与后续的可视化。

```python
model = TimeModel()
model.load_model("GCM_model.h5")
y_pred, y_test = model.predict()
with open("y_pred.json", "w") as file:
    y_pred_str = json.dumps(y_pred.tolist(), indent=4)
    file.write(y_pred_str)
with open("y_test.json", "w") as file:
    y_test_str = json.dumps(y_test.tolist(), indent=4)
    file.write(y_test_str)
```

### 4.2.2 评估结果&可视化

我们将所有的预测值导出并保存在本地项目的`y_pred.json`中，将所有的真实值导出并保存在本地项目的`y_test.json`中，在`pred_real_compare.py`中分析预测值和真实值之间的残差和残差率，公式如下：

```python
residual_percentage = [(t - p) / t * 100 if t != 0 else 0 for t, p in zip(test_values_sampled, pred_values_sampled)]

filtered_residuals = []
filtered_residual_percentage = []
for residual, percentage in zip(residuals_sampled, residual_percentage):
    if residual <= 20 and percentage <= 10 and percentage >= -10:
        filtered_residuals.append(residual)
        filtered_residual_percentage.append(percentage/100)
```

之后绘制残差与残差率图：

```python
# 绘制残差图
plt.figure(figsize=(10, 6))
plt.plot(range(len(filtered_residuals)), filtered_residuals, label='residuals', marker='d', color='r', linestyle='-')
plt.xlabel('Sample')
plt.ylabel('Residual')
plt.title('Residuals')
plt.legend()
plt.grid(True)
plt.show()

# 绘制残差率图
plt.figure(figsize=(10, 6))
plt.plot(range(len(filtered_residual_percentage)), filtered_residual_percentage, label='residual percentage', marker='o', color='b', linestyle='-')
plt.xlabel('Sample')
plt.ylabel('Residual Percentage')
plt.title('Residual Percentage')
plt.legend()
plt.grid(True)
plt.show()
```

可视化结果如下：

![Diff curve](./pic/Diff_Curve.png)
![Diff Ratio curve](./pic/Diff_Ratio_Curve.png)

## 4.3 基于均方误差（MSE）的评估

### 4.3.1 评估模型介绍

均方误差（MSE）是一种常用的评估预测模型性能的指标。它通过计算预测值与实际值之间的平方差的平均值来衡量模型的误差。由于误差被平方，`MSE`对较大的误差赋予了更高的权重，这使其特别适用于识别那些偶尔会产生较大偏差的模型。`MSE`特别适用于评估回归模型，提供了一个清晰的定量指标来衡量模型的预测精度。

### 4.3.2 评估过程

我们首先在`cal_MSE_MAE.py`中读取先前模型产出的`y_pred.json`数据与原始`y_test.json`数据，并初始化列表。

```python
test_values = []
pred_values = []

with open(TEST_PATH, 'r') as file:
    test_values = json.load(file)

with open(PRED_PATH, 'r') as file:
    pred_values = json.load(file)

    # 初始化列表
test_15min, test_30min, test_45min, test_60min = [], [], [], []
pred_15min, pred_30min, pred_45min, pred_60min = [], [], [], []
```

然后将15，30，45和60分钟的血糖浓度数据分别拆分：

```python
or i in range(len(test_values)):

    test_15min.append(test_values[i][0])
    test_30min.append(test_values[i][1])
    test_45min.append(test_values[i][2])
    test_60min.append(test_values[i][3])
    
    pred_15min.append(pred_values[i][0])
    pred_30min.append(pred_values[i][1])
    pred_45min.append(pred_values[i][2])
    pred_60min.append(pred_values[i][3])
```

最后计算`MSE`水平并打印结果：

```python
mse_15min = np.mean((np.array(test_15min) - np.array(pred_15min)) ** 2)
mse_30min = np.mean((np.array(test_30min) - np.array(pred_30min)) ** 2)
mse_45min = np.mean((np.array(test_45min) - np.array(pred_45min)) ** 2)
mse_60min = np.mean((np.array(test_60min) - np.array(pred_60min)) ** 2)
```

### 4.3.3 评估结果&可视化

均方误差（MSE）的评估结果如下：

![MAE_MSE result](./pic/MAE_MSE_Result.png)

