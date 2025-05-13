import os
import re

# 定义结果文件夹路径
results_dir = 'informer-simplify\\results'

# 遍历结果文件夹下的所有子文件夹
for folder in os.listdir(results_dir):
    folder_path = os.path.join(results_dir, folder)
    if os.path.isdir(folder_path):  # 确保是文件夹
        # 检查并重命名文件夹
        new_folder_name = folder
        
        # 使用正则表达式保留数字部分
        if 'lee' in folder:
            match = re.match(r'(.*lee)(\d+)', folder)
            if match:
                new_folder_name = f"{match.group(1)}{match.group(2)}"  # 保留lee和数字
        elif 'gelu' in folder:
            match = re.match(r'(.*gelu)(\d+)', folder)
            if match:
                new_folder_name = f"{match.group(1)}{match.group(2)}"  # 保留gelu和数字
        
        # 如果新文件夹名称不同，则重命名
        if new_folder_name != folder:
            new_folder_path = os.path.join(results_dir, new_folder_name)
            os.rename(folder_path, new_folder_path)
            print(f'重命名: {folder} -> {new_folder_name}')
