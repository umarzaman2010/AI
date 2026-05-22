# Lesson 01 Teacher Notes

## What Forward Diffusion Means

Forward diffusion is the process of gradually destroying a clean image by adding Gaussian noise.

This process is fixed. The neural network does not learn it.

The model only learns the reverse direction later.

## The Most Important Formula

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
```

Read it as:

```text
noisy image = some clean image + some random noise
```

At small timesteps:

- `alpha_bar_t` is close to 1.
- `sqrt(alpha_bar_t)` is large.
- `sqrt(1 - alpha_bar_t)` is small.
- The image still looks clean.

At large timesteps:

- `alpha_bar_t` is close to 0.
- `sqrt(alpha_bar_t)` is small.
- `sqrt(1 - alpha_bar_t)` is large.
- The image looks like noise.

## Beta, Alpha, Alpha Bar

`beta_t`:

```text
how much noise we add at step t
```

`alpha_t`:

```text
1 - beta_t
```

This is roughly how much signal remains after one step.

`alpha_bar_t`:

```text
alpha_1 * alpha_2 * ... * alpha_t
```

This is how much clean-image signal remains after many steps.

## Why Direct Sampling Is Useful

We do not need to run 500 noising steps to get `x_500`.

The formula lets us jump directly from `x_0` to any `x_t`.

That is important during training because each batch can randomly choose different timesteps.

## What To Notice In The Plots

In `01_noise_schedule.png`:

- `beta` slowly increases.
- `alpha` slowly decreases.
- `alpha_bar` drops strongly over time.

In `02_forward_diffusion_visual.png`:

- Early timesteps keep the shape visible.
- Middle timesteps partially hide it.
- Late timesteps are mostly random noise.

## Checkpoint Answers You Should Build Toward

- The forward process is fixed, not learned.
- `alpha_bar_t` decreases because we multiply many values below 1.
- At large `t`, most original image information is gone.
- Direct sampling works because the repeated Gaussian noising process has a closed-form expression.

