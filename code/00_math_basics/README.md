# Lesson 00 Code: Foundations

Run these scripts from the workspace root:

```powershell
python code/00_math_basics/01_tensor_and_noise.py
python code/00_math_basics/02_gaussian_histogram.py
python code/00_math_basics/03_tiny_training_loop.py
python code/00_math_basics/04_spread_and_scaling.py
python code/00_math_basics/05_statistics_basics.py
python code/00_math_basics/06_mean_balance_point.py
python code/00_math_basics/07_mean_balance_outlier_examples.py
python code/00_math_basics/08_standard_deviation_spread.py
python code/00_math_basics/09_std_positions.py
```

Outputs are saved in:

```text
outputs/foundations/
```

## What You Should Learn

- Images can be represented as tensors.
- Gaussian noise is random values sampled from a normal distribution.
- A neural network learns by reducing a loss.
- A training loop has the same basic shape in almost every deep learning project.
- Mean, median, mode, variance, and standard deviation summarize values in different ways.
- Variance describes noise spread, and multiplying noise changes its variance by the multiplier squared.
