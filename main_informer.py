from exp.exp_config import InformerConfig
from exp.exp_informer import Exp_Informer
import torch

def main():
    config = InformerConfig()
    config.use_gpu = torch.cuda.is_available()

    exp = Exp_Informer(config)
    exp.run_full_experiment(include_baseline=True)  # 可设为 False 跳过 baseline

if __name__ == "__main__":
    main()