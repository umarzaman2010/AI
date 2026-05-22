# Lesson 01: Forward Diffusion

## Goal

Understand how a clean image becomes noisy over many timesteps.

## Concept

Forward diffusion is not learned. It is a fixed process that gradually adds Gaussian noise to real data.

At timestep 0, the image is clean.

At the final timestep, the image is almost pure noise.

## Key Variables

```text
x_0 = clean image
x_t = noisy image at timestep t
epsilon = Gaussian noise
beta_t = amount of noise added at step t
alpha_t = 1 - beta_t
alpha_bar_t = product of alpha values up to t
```

## Important Formula

Instead of adding noise one step at a time, we can directly sample any noisy timestep:

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
```

This is one of the most important formulas in DDPM.

## Teacher Explanation

The formula mixes two things:

- A faded version of the original image.
- A growing amount of random noise.

When `alpha_bar_t` is close to 1, the image remains clear.

When `alpha_bar_t` is close to 0, the image disappears into noise.

## Practical Tasks

Create code in `code/01_diffusion_from_scratch/`.

I have added runnable scripts for this lesson:

```powershell
python code/01_diffusion_from_scratch/01_noise_schedule.py
python code/01_diffusion_from_scratch/02_forward_diffusion_visual.py
```

Read the teacher notes here:

```text
notes/lesson_01_teacher_notes.md
```

The generated plots are saved here:

```text
outputs/forward_diffusion/
```

Task 1:

- Implement a linear beta schedule.
- Compute alpha.
- Compute alpha_bar.

Task 2:

- Load one MNIST image.
- Show it at several timesteps.
- Save the grid to `outputs/noising_process/`.

Task 3:

- Plot beta, alpha, and alpha_bar over time.

## Checkpoint Questions

- Is the forward process learned or fixed?
- Why does `alpha_bar_t` decrease over time?
- What happens when `t` is very large?
- Why can we sample `x_t` directly instead of looping from `x_0` to `x_t`?
