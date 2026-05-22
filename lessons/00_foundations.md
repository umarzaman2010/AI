# Lesson 00: Foundations

## Goal

Before diffusion, you need enough PyTorch, probability, and neural network basics to understand what the model is doing.

## Concepts

Diffusion models are built from simple pieces:

- Images are tensors.
- Noise is usually sampled from a Gaussian distribution.
- Neural networks learn by minimizing a loss.
- Backpropagation updates model weights.
- A training loop repeatedly samples data, computes loss, and updates parameters.

## Minimum Math

Gaussian noise:

```text
epsilon ~ N(0, I)
```

This means every pixel receives random noise from a standard normal distribution.

Mean:

```text
average value
```

Variance:

```text
how spread out values are
```

Standard normal:

```text
mean = 0, variance = 1
```

## Practical Tasks

Create scripts or notebooks in `code/00_math_basics/`.

I have added runnable scripts for this lesson:

```powershell
python code/00_math_basics/01_tensor_and_noise.py
python code/00_math_basics/02_gaussian_histogram.py
python code/00_math_basics/03_tiny_training_loop.py
```

Read the teacher notes here:

```text
notes/lesson_00_teacher_notes.md
```

The generated plots are saved here:

```text
outputs/foundations/
```

Task 1:

- Create a tensor of zeros shaped like a grayscale image: `[1, 28, 28]`.
- Create Gaussian noise with the same shape.
- Add the noise to the image.
- Plot the result.

Task 2:

- Sample 10,000 values from a standard normal distribution.
- Plot a histogram.
- Compute mean and standard deviation.

Task 3:

- Build a tiny PyTorch model with `nn.Linear`.
- Train it on a simple toy function like `y = 2x + 1`.

## Checkpoint Questions

- What is a tensor?
- What shape does a grayscale MNIST image usually have?
- What does `torch.randn_like(x)` do?
- Why do we need a loss function?
- What does `optimizer.step()` change?
