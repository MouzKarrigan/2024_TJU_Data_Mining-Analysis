import matplotlib.pyplot as plt
import json
import numpy as np

TEST_PATH = './y_test.json'
PRED_PATH = './y_pred.json'

# 读取数据
test_values = []
pred_values = []

with open(TEST_PATH, 'r') as file:
    test_values = json.load(file)

with open(PRED_PATH, 'r') as file:
    pred_values = json.load(file)

    # 初始化列表
test_15min, test_30min, test_45min, test_60min = [], [], [], []
pred_15min, pred_30min, pred_45min, pred_60min = [], [], [], []

# 拆分数据
for i in range(len(test_values)):

    test_15min.append(test_values[i][0])
    test_30min.append(test_values[i][1])
    test_45min.append(test_values[i][2])
    test_60min.append(test_values[i][3])
    
    pred_15min.append(pred_values[i][0])
    pred_30min.append(pred_values[i][1])
    pred_45min.append(pred_values[i][2])
    pred_60min.append(pred_values[i][3])

# 计算MSE
mse_15min = np.mean((np.array(test_15min) - np.array(pred_15min)) ** 2)
mse_30min = np.mean((np.array(test_30min) - np.array(pred_30min)) ** 2)
mse_45min = np.mean((np.array(test_45min) - np.array(pred_45min)) ** 2)
mse_60min = np.mean((np.array(test_60min) - np.array(pred_60min)) ** 2)

# 计算MAE
mae_15min = np.mean(np.abs(np.array(test_15min) - np.array(pred_15min)))
mae_30min = np.mean(np.abs(np.array(test_30min) - np.array(pred_30min)))
mae_45min = np.mean(np.abs(np.array(test_45min) - np.array(pred_45min)))
mae_60min = np.mean(np.abs(np.array(test_60min) - np.array(pred_60min)))

# 打印结果

print('-----------------------------------')
print('MSE for each time period:\n')

print(f'MSE for 15 min: {mse_15min}')
print(f'MSE for 30 min: {mse_30min}')
print(f'MSE for 45 min: {mse_45min}')
print(f'MSE for 60 min: {mse_60min}')
print(f'MSE for all: {(mse_15min + mse_30min + mse_45min + mse_60min) / 4}')

print('-----------------------------------')
print('MAE for each time period:\n')

print(f'MAE for 15 min: {mae_15min}')
print(f'MAE for 30 min: {mae_30min}')
print(f'MAE for 45 min: {mae_45min}')
print(f'MAE for 60 min: {mae_60min}')
print(f'MAE for all: {(mae_15min + mae_30min + mae_45min + mae_60min) / 4}')
print('-----------------------------------')




