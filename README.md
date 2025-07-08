# Fibonacci Model Showdown

This repo explores **three different approaches** to forecasting the Fibonacci sequence:

| Model | Library | Key Idea | Typical Training Time* |
|-------|---------|----------|-------------------------|
| **Ordinary Least‑Squares (OLS)** | *scikit‑learn* | Classic linear regression on `[Fₜ, Fₜ₋₁] → Fₜ₊₁` | **\< 1 ms** |
| **RNN + Projection Layer** | *PyTorch* | Scalar input first passes through a small fully‑connected layer (proj = 4) before a 1‑layer RNN (hidden = 10) | 1‑2 s |
| **1‑D CNN** | *PyTorch* | Two Conv1d layers + global pooling learn the “add two numbers” rule in parallel | 1‑2 s |

---

## Why include a plain OLS model?

For deterministic linear recurrences like Fibonacci, **OLS is by far the best tool**—it nails the weights ≈ 1, 1, intercept ≈ 0 instantly and predicts perfectly.  
Deep‑learning models are great teaching exercises and become essential for **non‑linear or noisy** sequences, but simple problems often bow to classic statistics.

---

## Repo Structure

├── scripts/ # PyTorch CNN implementation, # PyTorch RNN w/ projection layer, # OLS baseline in plain NumPy + scikit‑learn

├── requirements.txt # pip install -r requirements.txt

└── README.md

## Contributing
If you’d like to contribute or report any issues, please open a Pull Request or file an Issue on this repository.
## Acknowledgments
I would like to thank Dr. Rahul Makhijani for their guidance and for allowing me to share this problem. Their insights and support were invaluable in completing this assignment successfully.
