import os
import pandas as pd
import json

# 统计有多少种药物并编号
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

agents_list_sorted = sorted(agents_list, key=len, reverse=True)

agent_str = json.dumps(agents_list_sorted, indent=2)

with open('agents_info.json', 'w') as file:
    file.write(agent_str)



# 每个人用哪些药物