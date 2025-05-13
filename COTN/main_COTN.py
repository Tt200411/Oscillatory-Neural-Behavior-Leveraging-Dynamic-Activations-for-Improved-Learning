from exp.exp_config import InformerConfig
from exp.exp_informer import Exp_Informer
import torch

def main():
    # 创建配置
    config = InformerConfig()
    
    # 根据是否有GPU调整设备设置
    config.use_gpu = True if torch.cuda.is_available() and config.use_gpu else False
    
    # 创建实验
    exp = Exp_Informer(config)
    
    # 训练所有的lee_type（从1到8）
    for i in range(1,101):
            # 设置当前的lee_type
        config.lee_type = 2
        print("start training" + str(i), str(config.lee_type))
        setting = f'informer_{config.data}_lee{config.lee_type}'
                
                # 开始训练
        print('>>>>>>>开始训练 : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
                
                # 测试
        print('>>>>>>>测试 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
            
            # 如果需要进行预测，可以取消注释以下代码
            # print('>>>>>>>预测 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # exp.predict(setting, True)

if __name__ == "__main__":
    main()
