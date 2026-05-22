# Lesson 02: Noise Prediction

## Goal

Understand what the neural network learns.

## Concept

In the basic DDPM setup, the model does not directly predict the clean image.

It predicts the noise that was added.

Training example:

```text
clean image x_0
random timestep t
random noise epsilon
noisy image x_t
model predicts epsilon
loss = MSE(predicted_epsilon, true_epsilon)
```

## Why Predict Noise?

Predicting noise gives a simple and stable training target.

The model learns:

```text
Given this noisy image and this timestep, what noise should be removed?
```

## Practical Tasks

Create code in `code/01_diffusion_from_scratch/` or `code/03_ddpm/`.

Task 1:

- Create `x_t` from `x_0`.
- Store the exact noise `epsilon` used.
- Train a small CNN to predict `epsilon`.

Task 2:

- Visualize true noise and predicted noise.
- Track MSE loss.

Task 3:

- Test the model at early, middle, and late timesteps.

## Checkpoint Questions

- What is the model input?
- What is the model target?
- Why must the model know the timestep?
- Why is MSE a reasonable loss here?

