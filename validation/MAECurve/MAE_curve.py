import matplotlib.pyplot as plt

# 读取数据
epochs = []
loss = []
mae = []

LOG_PATH = './model_info.log'

with open(LOG_PATH, 'r') as file:
    for line in file:
        if line.startswith('Epoch'):
            parts = line.split(',')
            epochs.append(int(parts[0].split(':')[0].split()[1]))
            loss.append(float(parts[0].split('=')[1]))
            mae.append(float(parts[1].split('=')[1]))

# 绘制MAE曲线图
plt.figure(figsize=(10, 6))
plt.plot(epochs, mae, marker='o', linestyle='-')
plt.title('MAE Curve')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.grid(True)
plt.show()
