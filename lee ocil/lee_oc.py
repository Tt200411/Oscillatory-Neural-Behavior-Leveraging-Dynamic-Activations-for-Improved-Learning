import torch
import torch.nn as nn
import torch.nn.functional as F

class LeeOscillator(nn.Module):
    def __init__(self):
        super(LeeOscillator, self).__init__()
        # 通用参数
        self.N = 100          # 时间步数
        self.e = 0.001        # 外部刺激大小
        self.k = 500          # K 值
        
        # 生成刺激范围张量 (-1 到 1)
        self.register_buffer('i_values', torch.arange(-1, 1, 0.001))
    
    def _tanh(self, x):
        """通用tanh激活函数"""
        return torch.tanh(5 * x)
    
    def type1(self, x):
        """第一类Lee振荡器"""
        # 参数设置
        a1, a2, a3, a4 = 0, 5.0, 5.0, 1.0
        b1, b2, b3, b4 = 0, -1.0, 1.0, 0
        xi_E, xi_I = 0, 0.0
        
        return self._run_oscillator(x, a1, a2, a3, a4, b1, b2, b3, b4, xi_E, xi_I)
    
    def type2(self, x):
        """第二类Lee振荡器"""
        a1, a2, a3, a4 = 0.5, 0.55, 0.55, -0.5
        b1, b2, b3, b4 = 0.5, -0.55, -0.55, -0.5
        xi_E, xi_I = 0, 0.0
        
        return self._run_oscillator(x, a1, a2, a3, a4, b1, b2, b3, b4, xi_E, xi_I)
    
    def type3(self, x):
        """第三类Lee振荡器"""
        a1, a2, a3, a4 = -5, 5, 5, -5
        b1, b2, b3, b4 = 1, -1, -1, 1
        xi_E, xi_I = 0, 0
        
        return self._run_oscillator(x, a1, a2, a3, a4, b1, b2, b3, b4, xi_E, xi_I)
    
    def type4(self, x):
        """第四类Lee振荡器"""
        a1, a2, a3, a4 = 1, 1, 1, -1
        b1, b2, b3, b4 = -1, -1, -1, 1
        xi_E, xi_I = 0, 0
        
        return self._run_oscillator(x, a1, a2, a3, a4, b1, b2, b3, b4, xi_E, xi_I)
    
    def type5(self, x):
        """第五类Lee振荡器"""
        a1, a2, a3, a4 = 5, -5, -5, 5
        b1, b2, b3, b4 = -1, 1, 1, -1
        xi_E, xi_I = 0, 0
        
        return self._run_oscillator(x, a1, a2, a3, a4, b1, b2, b3, b4, xi_E, xi_I)
    
    def type6(self, x):
        """第六类Lee振荡器"""
        a1, a2, a3, a4 = -1, -1, -1, 1
        b1, b2, b3, b4 = 1, 1, 1, -1
        xi_E, xi_I = 0, 0
        
        return self._run_oscillator(x, a1, a2, a3, a4, b1, b2, b3, b4, xi_E, xi_I)
    
    def type7(self, x):
        """第七类Lee振荡器"""
        a1, a2, a3, a4 = 1, -1, -1, 1
        b1, b2, b3, b4 = -1, 1, 1, -1
        xi_E, xi_I = 0, 0
        
        return self._run_oscillator(x, a1, a2, a3, a4, b1, b2, b3, b4, xi_E, xi_I)
    
    def type8(self, x):
        """第八类Lee振荡器"""
        a1, a2, a3, a4 = -1, 1, 1, -1
        b1, b2, b3, b4 = 1, -1, -1, 1
        xi_E, xi_I = 0, 0
        
        return self._run_oscillator(x, a1, a2, a3, a4, b1, b2, b3, b4, xi_E, xi_I)

    def _run_oscillator(self, x, a1, a2, a3, a4, b1, b2, b3, b4, xi_E, xi_I):
        """运行Lee振荡器的核心函数 - PyTorch版本"""
        # 保存原始形状
        original_shape = x.shape
        device = x.device
        
        # 将输入展平
        x = x.view(-1)
        
        # 初始化状态张量
        E = torch.zeros(self.N, device=device)
        I = torch.zeros(self.N, device=device)
        LORS = torch.zeros(self.N, device=device)
        Ω = torch.zeros(self.N, device=device)
        
        # 设置初始条件
        E[0] = 0.2
        LORS[0] = 0.2
        Ω[0] = 0
        
        # 对每个输入值运行振荡器
        results = torch.zeros_like(x)
        for idx, i in enumerate(x):
            # 计算外部刺激
            sim = i + self.e * torch.sign(i)
            
            # 运行振荡器
            for t in range(self.N - 1):
                E[t + 1] = self._tanh(a1 * LORS[t] + a2 * E[t] - a3 * I[t] + a4 * sim - xi_E)
                I[t + 1] = self._tanh(b1 * LORS[t] - b2 * E[t] - b3 * I[t] + b4 * sim - xi_I)
                Ω[t + 1] = self._tanh(sim)
                
                LORS[t + 1] = (E[t + 1] - I[t + 1]) * torch.exp(-self.k * sim * sim) + Ω[t + 1]
            
            # 存储结果
            results[idx] = LORS[-1]
        
        # 恢复原始形状
        return results.view(original_shape)

    def forward(self, x, oscillator_type=1):
        """前向传播函数，方便在nn.Module中使用"""
        oscillator_funcs = {
            1: self.type1,
            2: self.type2,
            3: self.type3,
            4: self.type4,
            5: self.type5,
            6: self.type6,
            7: self.type7,
            8: self.type8
        }
        return oscillator_funcs[oscillator_type](x)