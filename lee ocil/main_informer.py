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
    
    if config.lee_grid_search:
        print('>>>>>>>开始Lee振荡器网格搜索>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        best_config = exp.grid_search()
        
        # 使用最佳配置进行最终训练和测试
        config.encoder_lee_types = list(best_config['encoder_types'])
        config.decoder_lee_types = list(best_config['decoder_types'])
        exp = Exp_Informer(config)  # 重新创建实验
        setting = best_config['setting']
    else:
        setting = f'informer_ETTh1_ft{config.features}_sl{config.seq_len}_ll{config.label_len}_pl{config.pred_len}_dm{config.d_model}_nh{config.n_heads}_el{config.e_layers}_dl{config.d_layers}_df{config.d_ff}_fc{config.factor}_eb{config.freq}_dt{config.distil}_test'
        
        # 训练
        print('>>>>>>>开始训练 : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
    
    # 测试
    print('>>>>>>>测试 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)
    
    # 预测(如果需要)
    # print('>>>>>>>预测 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    # exp.predict(setting, True)

if __name__ == "__main__":
    main()
