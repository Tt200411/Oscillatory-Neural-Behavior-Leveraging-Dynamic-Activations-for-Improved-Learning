import torch
import torch.nn as nn

class LeeOscillator(nn.Module):
    def __init__(self):
        super().__init__()
        # 模拟总步数（包括 t=0 初始状态），实际使用 t=1~100
        self.N = 101  # 共101步，后续会丢弃第0步
        # 如果不需要全范围刺激点，这行可以删
        self.register_buffer('i_values', torch.arange(-1, 1, 0.002))

        # ——— 基础参数字典，共 8 种 base ———
        self.params = {
            1: (0.0, 5.0, 5.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 5.0, 0.001, 500),
            2: (0.5, 0.55, 0.55, -0.5, 0.5, -0.55, -0.55, -0.5, 0.0, 0.0, 1.0, 0.001, 50),
            3: (0.5, 0.6, 0.55, 0.5, -0.5, -0.6, -0.55, 0.5, 0.0, 0.0, 1.0, 0.001, 50),
            4: (-0.5, 0.55, 0.55, -0.5, -0.5, -0.55, -0.55, 0.5, 0.0, 0.0, 1.0, 0.001, 50),
            5: (-0.9, 0.9, 0.9, -0.9, 0.9, -0.9, -0.9, 0.9, 0.0, 0.0, 1.0, 0.001, 50),
            6: (-0.9, 0.9, 0.9, -0.9, 0.9, -0.9, -0.9, 0.9, 0.0, 0.0, 1.0, 0.001, 300),
            7: (-5.0, 5.0, 5.0, -5.0, 1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.001, 50),
            8: (-5.0, 5.0, 5.0, -5.0, 1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.001, 300),
        }

        # ——— 动态创建 8 个激活函数 type1…type8 ———
        for base in sorted(self.params.keys()):
            setattr(self, f"type{base}", self._make_pooled_fn(base))

        self.num_types = len(self.params)

    def _make_pooled_fn(self, base: int):
        """
        返回一个 fn(x)，它会：
          1) 跑完整的 100 步振荡器（t=1~100），得到 shape [..., 100] 的时间序列；
          2) 在时间维度上做 max pooling，输出 shape 与 x 相同的张量。
        """
        def fn(x):
            full = self._run_full(x, base)   # 返回 [..., 101]
            seq = full[..., 1:]              # 丢弃 t=0，保留 t=1~100
            return seq.max(dim=-1).values
        return fn

    def _run_full(self, x: torch.Tensor, base: int) -> torch.Tensor:
        """
        对输入 x（任意形状），跑出 100 步 LORS 序列，返回 shape [..., 101]
        """
        a1, a2, a3, a4, \
        b1, b2, b3, b4, \
        xi_E, xi_I, mu, e, k = self.params[base]

        orig_shape = x.shape
        device = x.device
        x_flat = x.view(-1)
        B = x_flat.size(0)

        results = torch.zeros(B, self.N, device=device)

        for i, inp in enumerate(x_flat):
            E = torch.zeros(self.N, device=device)
            I = torch.zeros(self.N, device=device)
            LORS = torch.zeros(self.N, device=device)
            Ω = torch.zeros(self.N, device=device)

            # 初始化 t=0
            E[0], LORS[0], Ω[0] = 0.2, 0.2, 0.0
            sim0 = inp + e * torch.sign(inp)

            for t in range(self.N - 1):
                E[t+1]    = torch.tanh(mu * (a1 * LORS[t] + a2 * E[t] - a3 * I[t] + a4 * sim0 - xi_E))
                I[t+1]    = torch.tanh(mu * (b1 * LORS[t] - b2 * E[t] - b3 * I[t] + b4 * sim0 - xi_I))
                Ω[t+1]    = torch.tanh(mu * sim0)
                LORS[t+1] = (E[t+1] - I[t+1]) * torch.exp(-k * sim0 * sim0) + Ω[t+1]

            results[i] = LORS

        return results.view(*orig_shape, self.N)

    def forward(self, x: torch.Tensor, oscillator_type: int = 1) -> torch.Tensor:
        """
        oscillator_type ∈ [1,8]，输出对 100 个时间步（t=1~100）做了最大池化后的激活值，shape 与 x 相同。
        """
        if not (1 <= oscillator_type <= self.num_types):
            raise ValueError(f"oscillator_type must be in [1, {self.num_types}], got {oscillator_type}")
        return getattr(self, f"type{oscillator_type}")(x)
