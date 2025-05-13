ReadME
# Lee-Ocil: Oscillatory Activation for Robust Time Series Forecasting under Extreme Conditions

## 🔍 Overview

This repository presents **Lee-Ocil**, a novel oscillatory activation function designed for time series forecasting under extreme, volatile, or noisy conditions. Inspired by the behavior of physical oscillators, Lee-Ocil introduces controlled randomness to the training process, enhancing both model robustness and generalization across diverse forecasting tasks.

We integrate Lee-Ocil into a Transformer-based forecasting framework and evaluate its performance across over 200 experiments on benchmark and real-world datasets, including the ETT series, power market data, and A-share financial indices.

---

## 🌟 Features

- **Oscillatory Activation Function**: Mimics chaotic oscillators to introduce controlled perturbations during training.
- **Uncertainty-Aware**: Enhances exploration of solution space under volatile or low-signal-to-noise conditions.
- **GELU Pretraining Strategy**: Enables efficient fine-tuning from pretrained GELU models to accelerate convergence.
- **Broad Benchmark Testing**: Validated on ETTh1, ETTh2, ETTm1, ETTm2, power market, and financial datasets.
- **Extreme Scenario Robustness**: Outperforms standard activations like GELU in >77% of experiments.

---

## 🧠 Key Innovation: Lee-Ocil Activation

Lee-Ocil is formulated as a parameterized sine-based activation function with trainable amplitude and frequency components:

```python
y = x * torch.sin(alpha * x + beta)

```

- `alpha` and `beta` are learnable parameters controlling frequency and phase shift.
- This non-monotonic, oscillatory behavior introduces mild chaos that prevents early convergence to sharp minima, improving generalization.

---

## 🧪 Experimental Setup

- **Framework**: PyTorch 2.1.0 + CUDA 11.8
- **Environment**: Windows 11, Intel i7-11800H, NVIDIA RTX 3060
- **Backbone**: Hybrid Autoencoder Transformer
- **Runs**: 200+ independent training trials (>8 hours total)
- **Evaluation**: MSE, MAE, convergence time, uncertainty distribution

---

## 📊 Results Summary

- **Performance Boost**: Up to **21% accuracy improvement** in long-term forecasting vs. GELU.
- **Stability**: Lee-Ocil outperformed GELU in **77%** of trials.
- **Robustness**: Strong resilience on high-frequency/noisy data (e.g., ETTm2).
- **Efficiency**: GELU-to-Lee-Ocil fine-tuning reduces training time by **35%+**.

---

## 📦 Installation

```bash
git clone <https://github.com/your-username/Lee-Ocil-Extreme-Forecasting.git>
cd Lee-Ocil-Extreme-Forecasting
pip install -r requirements.txt

```

---

## 🚀 Usage

### Train from scratch with Lee-Ocil:

```bash
python train.py --activation lee-ocil --dataset ETTh1

```

### Fine-tune from GELU pretrained weights:

```bash
python train.py --activation lee-ocil --pretrained_path ./checkpoints/gelu_etth1.pth

```

---

## 📁 Project Structure

```
├── models/
│   └── activation.py     # Implementation of Lee-Ocil
│   └── transformer.py    # Backbone forecasting model
├── datasets/
├── scripts/
├── results/
├── train.py
├── evaluate.py
├── README.md
└── requirements.txt

```

---

## 📈 Visual Results

| Dataset | Baseline (GELU) | Lee-Ocil (Ours) | ∆ MSE | ∆ MAE |
| --- | --- | --- | --- | --- |
| ETTh1 | 0.413 | 0.325 | -0.088 | -0.041 |
| ETTm2 | 0.587 | 0.422 | -0.165 | -0.069 |
| Power | 0.356 | 0.278 | -0.078 | -0.032 |
| Stock | 0.441 | 0.352 | -0.089 | -0.043 |

---

## 🧩 Citation

If you find this project useful in your research, please cite:

```
@article{leeocil2025,
  title={Lee-Ocil: Oscillatory Activation for Robust Time Series Forecasting under Extreme Conditions},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}

```

---

## 🤝 Contribution

We welcome contributions from the community! Please feel free to open issues or submit pull requests if you:

- Found a bug
- Have ideas for improvements
- Want to test Lee-Ocil in new domains (e.g., weather, traffic, health data)

---

## 📬 Contact

For questions, collaborations, or feedback, please reach out to:

**📧 [your.email@example.com](mailto:your.email@example.com)**

---

## 🧭 Roadmap

- [x]  Lee-Ocil activation module
- [x]  Transformer integration
- [x]  GELU pretraining + Lee-Ocil fine-tuning
- [ ]  JAX version
- [ ]  Real-time forecasting deployment
- [ ]  Cross-domain evaluation (weather, IoT, smart grids)

---

> Lee-Ocil helps models hear the rhythm in chaos — a new approach to dancing with uncertainty.
> 

```

```