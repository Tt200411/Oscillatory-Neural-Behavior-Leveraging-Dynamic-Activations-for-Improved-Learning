def calculate_new_error(input_data):
    # 将输入数据按行分割并转换为浮点数
    numbers = [float(x.strip()) for x in input_data.strip().split('\n') if x.strip()]
    
    # 检查数据是否能被4整除
    if len(numbers) % 4 != 0:
        raise ValueError("输入数据必须是4的倍数")
    
    results = []
    # 每4个数据为一组进行处理
    for i in range(0, len(numbers), 4):
        group = numbers[i:i+4]
        old_error = group[0]  # 旧误差
        improvement_ratio = group[1]  # 提升比例
        
        # 计算新误差
        new_error = old_error * (1 - improvement_ratio)
        results.append(new_error)
    
    return results

def main():
    print("请输入数据（每组4个数，用换行分隔）：")
    input_data = ""
    try:
        while True:
            line = input()
            if not line:  # 空行表示输入结束
                break
            input_data += line + "\n"
    except EOFError:
        pass
    
    try:
        new_errors = calculate_new_error(input_data)
        print("\n计算结果：")
        for i, error in enumerate(new_errors, 1):
            print(f"第{i}组新误差: {error:.3f}")
    except ValueError as e:
        print(f"错误：{str(e)}")
    except Exception as e:
        print(f"发生错误：{str(e)}")

if __name__ == "__main__":
    main()