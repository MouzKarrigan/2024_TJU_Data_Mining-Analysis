import matplotlib.pyplot as plt
import json



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

# 计算差距
differences_sampled = [abs(t - p) for t, p in zip(test_values_sampled, pred_values_sampled)]

# 绘制曲线图
plt.figure(figsize=(10, 6))
plt.plot(range(len(differences_sampled)), differences_sampled, label='Differences', marker='d', color='r', linestyle='-')

# 添加标签和标题
plt.xlabel('Sample')
plt.ylabel('Difference')
plt.title('Differences of Test Values and pred Values')
plt.legend()
plt.grid(True)
plt.show()