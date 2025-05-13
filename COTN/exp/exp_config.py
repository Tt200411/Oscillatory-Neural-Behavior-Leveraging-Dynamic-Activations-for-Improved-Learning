class InformerConfig:
    def __init__(self):
        # 数据参数
        self.seq_len = 96  # 输入序列长度
        self.label_len = 48  # 解码器开始令牌长度
        self.pred_len = 24  # 预测序列长度
        
        # 模型参数
        self.enc_in = 7  # 编码器输入大小
        self.dec_in = 7  # 解码器输入大小
        self.c_out = 7  # 输出大小
        self.d_model = 512  # 模型维度
        self.n_heads = 8  # 注意力头数
        self.e_layers = 2  # 编码器层数
        self.d_layers = 1  # 解码器层数
        self.d_ff = 2048  # 前馈网络维度
        self.factor = 5  # ProbSparse注意力因子
        self.dropout = 0.05  # dropout率
        self.attn = 'prob'  # 注意力类型 ('prob' 或 'full')
        self.s_layers = [3,2,1]  # informerstack的编码器层数配置
        
        # 训练参数
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.train_epochs = 6
        self.patience = 3  # 早停耐心值
        self.num_workers = 0  # 数据加载线程数
        self.lradj = 'type1'  # 学习率调整类型
        
        # 其他参数
        self.output_attention = False  # 是否输出注意力
        self.mix = True  # 是否使用混合注意力
        self.padding = 0  # padding类型
        self.freq = 'h'  # 时间特征编码频率
        self.distil = True  # 是否使用蒸馏
        self.embed = 'fixed'  # 嵌入类型
        self.cols = None  # 使用的列
        
        # 模型类型
        self.model = 'informer'  # 模型类型：informer/informerstack
        self.data = 'ETTh1'  # 数据集名称
        self.detail_freq = 'h'  # 预测时的频率
        self.checkpoints = './checkpoints/'  # 检查点保存路径
        self.use_amp = False  # 是否使用混合精度训练
        
        # 设备参数
        self.use_gpu = True
        self.use_multi_gpu = False
        self.device = 'cuda:0' # GPU设备ID
        self.gpu = 0
        self.devices = '0'  # GPU设备ID

        # 数据集参数
        self.root_path = 'ETT-small'
        self.data = 'ETTh1'                # 默认数据集
        self.data_path = f'{self.data}.csv' # 动态生成路径
        self.features = 'M'  # 预测任务类型
        self.target = 'OT'  # 目标特征
        self.inverse = False  # 是否反转输出数据
        
        # 激活函数参数
        self.activation = 'Lee'  # 激活函数类型：'gelu', 'relu', 'lee'
        self.lee_oscillator = True  # 是否使用Lee振荡器作为激活函数
        self.lee_type = 3  # 使用第几类Lee振荡器 (1-8)
        
        # 其他参数保持不变 ... 