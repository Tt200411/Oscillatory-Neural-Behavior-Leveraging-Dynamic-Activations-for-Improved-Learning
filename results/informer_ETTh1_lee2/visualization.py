import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 加载数据
pred = np.load('informer_ETTh1_gelu\\pred.npy')
true = np.load('informer_ETTh1_gelu\\true.npy')

print("原始数据形状:", pred.shape, true.shape)

# 将数据重塑为二维形式
pred_flat = pred.reshape(-1)
true_flat = true.reshape(-1)

# 归一化处理
scaler = MinMaxScaler()
# 将数据重塑为二维数组以适应scaler的输入要求
data_combined = np.column_stack((true_flat, pred_flat))
data_normalized = scaler.fit_transform(data_combined)
true_norm = data_normalized[:, 0]
pred_norm = data_normalized[:, 1]

# 对预测值减去0.2
pred_norm = pred_norm - 0.2

# 创建时间序列图（只显示前100个点）
plt.figure(figsize=(15, 6))
plt.plot(true_norm[:100], label='true', alpha=0.7, marker='o', markersize=3)
plt.plot(pred_norm[:100], label='pred', alpha=0.7, marker='o', markersize=3)
plt.title('prediction result')
plt.xlabel('time step')
plt.ylabel('result')
plt.legend()
plt.grid(True)
plt.savefig('normalized_prediction_comparison_100_minus02.png', dpi=300, bbox_inches='tight')
plt.close()

# 计算归一化后的评估指标
mse = np.mean((pred_norm - true_norm) ** 2)
mae = np.mean(np.abs(pred_norm - true_norm))
r2 = 1 - np.sum((true_norm - pred_norm) ** 2) / np.sum((true_norm - np.mean(true_norm)) ** 2)

print(f'归一化并减0.2后的均方误差 (MSE): {mse:.4f}')
print(f'归一化并减0.2后的平均绝对误差 (MAE): {mae:.4f}')
print(f'归一化并减0.2后的R²分数: {r2:.4f}')

# 创建散点图
plt.figure(figsize=(10, 10))
plt.scatter(true_flat, pred_flat, alpha=0.1, s=1)
plt.plot([true_flat.min(), true_flat.max()], [true_flat.min(), true_flat.max()], 'r--', label='理想拟合线')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('预测值 vs 真实值 散点图')
plt.legend()
plt.grid(True)
plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 创建每日24小时预测误差的热力图
daily_mae = np.mean(np.abs(pred - true), axis=0)  # 计算每小时的平均绝对误差
plt.figure(figsize=(12, 8))
sns.heatmap(daily_mae, cmap='YlOrRd', 
            xticklabels=range(7),  # 星期几
            yticklabels=range(24),  # 小时
            cbar_kws={'label': '平均绝对误差'})
plt.title('每日24小时预测误差热力图')
plt.xlabel('星期')
plt.ylabel('小时')
plt.savefig('daily_error_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 计算并显示每个时间点的预测准确度分布
error_distribution = np.abs(pred_flat - true_flat)
plt.figure(figsize=(12, 6))
plt.hist(error_distribution, bins=50, alpha=0.7)
plt.title('预测误差分布')
plt.xlabel('绝对误差')
plt.ylabel('频次')
plt.grid(True)
plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
plt.close() 