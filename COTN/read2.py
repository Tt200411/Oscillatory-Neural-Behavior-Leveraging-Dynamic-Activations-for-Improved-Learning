import os
import numpy as np

# —— 配置区 ——  
results_dir = r'C:\Users\18319\Desktop\chaotic-NN-bitcoin-prediction333\informer-simplify\results'
dataset     = 'ETTh1'  # 或 'ETTm1'  
baseline    = f"informer_{dataset}_gelu"
lee_models  = [f"informer_{dataset}_lee{i}" for i in range(1, 6)]
metric_names = ['mae', 'mse', 'rmse', 'mape', 'mspe']
# metrics.npy 的存储顺序假设是：[mae, mse, rmse, mape, mspe]
metric_idx   = {name: idx for idx, name in enumerate(metric_names)}

# —— 加载基准 ——  
baseline_path    = os.path.join(results_dir, baseline, 'metrics.npy')
baseline_metrics = np.load(baseline_path)  # 形状 (5,) ，[mae,mse,…,mspe]

# —— 加载 5 个 Lee 模型 ——  
lee_metrics = {}
for model in lee_models:
    path = os.path.join(results_dir, model, 'metrics.npy')
    lee_metrics[model] = np.load(path)

print(f"\n=== 数据集: {dataset} ===")
for m in metric_names:
    idx = metric_idx[m]
    base_val = baseline_metrics[idx]
    print(f"\n指标: {m} （基准: {base_val:.4f}）")

    # 计算 5 个 Lee 相对提升
    improvements = {}
    for model, metrics in lee_metrics.items():
        lee_val = metrics[idx]
        imp = (base_val - lee_val) / base_val * 100
        improvements[model] = imp
        print(f"  {model:20s}  -> {lee_val:.4f}，提升 {imp:6.2f}%")

    # 找出提升最多的那个
    best_model = max(improvements, key=improvements.get)
    best_imp   = improvements[best_model]
    print(f"  >>> 最佳 Lee 模型: {best_model}，提升 {best_imp:.2f}%")
