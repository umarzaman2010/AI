# Lesson 04: DDPM Training

## Goal

Train a complete basic DDPM on MNIST.

## Training Algorithm

For each batch:

```text
1. Sample clean images x_0.
2. Sample random timesteps t.
3. Sample Gaussian noise epsilon.
4. Create noisy images x_t.
5. Predict noise with U-Net: epsilon_theta(x_t, t).
6. Compute MSE loss.
7. Backpropagate and update weights.
```

## Important Detail

Each batch contains different timesteps. The model learns to denoise at all noise levels, not just one.

## Practical Tasks

Create code in `code/03_ddpm/`.

Task 1:

- Implement a `DiffusionSchedule` helper.
- Implement `q_sample(x_0, t, noise)`.

Task 2:

- Train U-Net on MNIST.
- Save loss values.
- Save checkpoints in `checkpoints/`.

Task 3:

- Save generated sample grids in `outputs/samples/`.

## Checkpoint Questions

- Why do we sample random timesteps during training?
- What is the difference between `x_0` and `x_t`?
- What does the loss compare?
- Why do we save checkpoints?

