import os
import numpy as np
import matplotlib.pyplot as plt

# 定义结果文件夹路径
results_dir = 'informer-simplify\\results'
metrics_data = {}

# 基准模型路径
baseline_model_path = 'informer-simplify\\results\informer_ETTh1_gelu'
baseline_metrics_file = os.path.join(baseline_model_path, 'metrics.npy')
baseline_metrics = np.load(baseline_metrics_file) if os.path.exists(baseline_metrics_file) else None

# 遍历结果文件夹下的所有子文件夹
for folder in os.listdir(results_dir):
    folder_path = os.path.join(results_dir, folder)
    if os.path.isdir(folder_path):  # 确保是文件夹
        metrics_file = os.path.join(folder_path, 'metrics.npy')
        if os.path.exists(metrics_file):  # 检查 metrics.npy 是否存在
            metrics = np.load(metrics_file)
            metrics_data[folder] = metrics  # 按文件夹命名存储数据
            print(f'{folder}: {metrics.shape}')  # 打印每个文件夹的 metrics 形状

# 分开比较 ETTm1 和 ETTh1 数据集
datasets = {'ETTm1': {}, 'ETTh1': {}}

for folder, metrics in metrics_data.items():
    if 'ETTm1' in folder:
        datasets['ETTm1'][folder] = metrics
    elif 'ETTh1' in folder:
        datasets['ETTh1'][folder] = metrics

# 比较不同参数的拟合效果
for dataset_name, dataset_metrics in datasets.items():
    print(f"\n比较数据集: {dataset_name}")
    best_models = {}
    for metric_name in ['mae', 'mse', 'rmse', 'mape', 'mspe']:
        plt.figure(figsize=(10, 5))
        best_value = float('inf') if metric_name in ['mae', 'mse', 'rmse'] else float('-inf')
        best_model = None

        for folder, metrics in dataset_metrics.items():
            plt.plot(metrics, label=folder)  # 绘制每个文件夹的指标
            
            # 找到最佳模型
            current_value = metrics[0]  # 假设第一个值是我们要比较的指标
            if (metric_name in ['mae', 'mse', 'rmse'] and current_value < best_value) or \
               (metric_name in ['mape', 'mspe'] and current_value > best_value):
                best_value = current_value
                best_model = folder

        # 计算提升百分比
        if best_model and baseline_metrics is not None:
            baseline_value = baseline_metrics[0]  # 基准模型的指标值
            improvement = ((baseline_value - best_value) / baseline_value) * 100
            
            # 如果提升百分比为负数，选择基准模型
            if improvement < 0:
                best_models[metric_name] = (baseline_model_path, 0)  # 选择基准模型
                print(f'最佳模型: {baseline_model_path}, 相对于基准模型的提升百分比: 0.00%')
            else:
                best_models[metric_name] = (best_model, improvement)
                print(f'最佳模型: {best_model}, 相对于基准模型的提升百分比: {improvement:.2f}%')

    # 打印每个指标的最佳模型和提升百分比
    for metric_name, (model, improvement) in best_models.items():
        print(f'{metric_name} 的最佳模型是 {model}，相对于基准模型的提升百分比为 {improvement:.2f}%')



