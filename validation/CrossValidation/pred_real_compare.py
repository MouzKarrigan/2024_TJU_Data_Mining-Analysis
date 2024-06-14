import matplotlib.pyplot as plt
import json
import time



TEST_PATH = './y_test.json'
PRED_PATH = './y_pred.json'

# 读取数据
test_values = []
pred_values = []

with open(TEST_PATH, 'r') as file:
    test_values = json.load(file)

with open(PRED_PATH, 'r') as file:
    pred_values = json.load(file)

# 将嵌套列表展开为一维列表
test_values_flat = [item for sublist in test_values for item in sublist]
pred_values_flat = [item for sublist in pred_values for item in sublist]

# 确保观测值和预测值的长度相同
assert len(test_values_flat) == len(pred_values_flat), "Observed and pred values must have the same length"

# 下采样（取每100个点中的一个点）
step = 1500
test_values_sampled = test_values_flat[::step]
pred_values_sampled = pred_values_flat[::step]

# 计算残差
residuals_sampled = [abs(t - p) for t, p in zip(test_values_sampled, pred_values_sampled)]

# 计算残差率
residual_percentage = [(t - p) / t * 100 if t != 0 else 0 for t, p in zip(test_values_sampled, pred_values_sampled)]

filtered_residuals = []
filtered_residual_percentage = []
for residual, percentage in zip(residuals_sampled, residual_percentage):
    if residual <= 20 and percentage <= 10 and percentage >= -10:
        filtered_residuals.append(residual)
        filtered_residual_percentage.append(percentage/100)

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