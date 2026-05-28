# Lesson 01: Forward Diffusion, From The Beginning

## Why This Lesson Exists

A diffusion model eventually learns to turn noise into an image. But first we define the opposite process:

```text
clean image -> slightly noisy image -> noisier image -> almost pure noise
```

This direction is called **forward diffusion** or the **forward noising process**.

It is not learned by a neural network. We design it ourselves so that training data can be corrupted in a controlled way.

## Start With One Pixel

Imagine one pixel has value:

```text
x_0 = 1.0
```

If we add a little random noise, it might become:

```text
x_1 = mostly the old pixel + a little random noise
```

For a whole image, this same idea happens at every pixel. The image tensor keeps the same shape; its values become less reliable.

## One Noising Step

Before reading the formula below, complete:

```text
lessons/00b_mean_variance_and_scaling.md
```

If the words `variance`, `standard deviation`, or `sqrt` still feel unfamiliar, do not push through them yet. They are the prerequisite for understanding this formula.

The mathematical rule for one step is:

```text
x_t = sqrt(alpha_t) * x_(t-1) + sqrt(beta_t) * epsilon_t
```

where:

```text
x_(t-1)   = image before this step
x_t       = image after this step
epsilon_t = fresh Gaussian noise
beta_t    = noise amount for this step
alpha_t   = 1 - beta_t
```

### Why Do We Need `beta_t`?

`beta_t` is the amount of random spread we choose to add at step `t`. In probability language, this spread amount is called **variance**.

For example:

```text
beta_t = 0.01
```

means we inject noise whose added variance is `0.01` at that step. It does not mean that `1%` of pixels are changed; every pixel receives random perturbation.

### Why Is `alpha_t = 1 - beta_t`?

We want a controlled mixture whose overall scale does not explode:

```text
some existing signal + some new noise
```

If `beta_t` is the new random spread assigned to noise, the remaining share is assigned to the existing image:

```text
alpha_t + beta_t = 1
```

therefore:

```text
alpha_t = 1 - beta_t
```

### Why Are There Square Roots?

Use the simpler word **multiplier** instead of amplitude.

`epsilon_t` begins as standard Gaussian noise with variance `1`. If we multiply it by a number `c`, its variance becomes `c^2`.

We want the noise term to have variance `beta_t`, so its multiplier must satisfy:

```text
c^2 = beta_t
c = sqrt(beta_t)
```

Therefore:

```text
image multiplier = sqrt(alpha_t)
noise multiplier = sqrt(beta_t)
```

Example:

```text
beta_t = 0.04
sqrt(beta_t) = 0.2
```

Multiplying standard noise by `0.2` creates noise with variance `0.04`.

## From One Step To Many Steps

After one step, the amount of original signal left is controlled by `alpha_1`.

After two steps, the first step has already kept only `alpha_1` of the signal, and the second step keeps only `alpha_2` of what remains:

```text
signal after two steps = alpha_1 * alpha_2
```

After three steps:

```text
signal after three steps = alpha_1 * alpha_2 * alpha_3
```

This is why we define:

```text
alpha_bar_t = alpha_1 * alpha_2 * ... * alpha_t
```

The bar means cumulative product over all steps up to `t`.

### Everyday Analogy

Suppose each step preserves `90%` of the original signal:

```text
after step 1: 0.90 remains
after step 2: 0.90 * 0.90 = 0.81 remains
after step 3: 0.90 * 0.90 * 0.90 = 0.729 remains
```

The amount remaining multiplies because every new step acts on what survived the previous steps.

## The Direct Formula

The repeated noising steps can be simplified into one formula:

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
```

Interpretation:

```text
noisy image at step t = remaining original image + accumulated noise
```

Why this is useful:

- To train, we do not need to calculate step 1, step 2, ..., step 500.
- We can choose a random step such as `t=500` and create that noisy image immediately.
- The model can efficiently train on many different noise levels.

## What Is A Noise Schedule?

A **schedule** is simply a pre-chosen list of noise amounts:

```text
beta_1, beta_2, beta_3, ..., beta_T
```

It answers this question:

```text
How much new noise do we add at each step?
```

We usually use many small steps, for example `T = 1000`, instead of destroying the image all at once. Small steps make the reverse learning problem more manageable.

### Linear Beta Schedule

A **linear beta schedule** means the noise amounts increase evenly from a small value to a larger value.

Example with only 5 steps:

```text
beta = [0.001, 0.006, 0.011, 0.016, 0.021]
```

The difference between neighboring beta values is constant.

In the practical script we use:

```text
beta_1    = 0.0001
beta_1000 = 0.02
```

and fill the values in between with `torch.linspace`.

The early steps add extremely little noise. Later steps add more new noise because the image is already damaged.

## Vocabulary Table

| Symbol | Plain meaning |
| --- | --- |
| `x_0` | Original clean image |
| `x_t` | Image after `t` noising steps |
| `epsilon` | Gaussian random noise |
| `T` | Total number of noising steps |
| `beta_t` | New noise variance added at step `t` |
| `alpha_t` | Signal variance preserved at one step: `1 - beta_t` |
| `alpha_bar_t` | Signal variance remaining after all steps from `1` to `t` |

## Practical Work

Use the code in `code/01_diffusion_from_scratch/`.

### Part A: One-Step And Repeated Signal Intuition

Run:

```powershell
python code/01_diffusion_from_scratch/00_why_alpha_bar_multiplies.py
```

Look for:

- A single `alpha` value below 1.
- The remaining signal decreasing as the same type of step is repeated.
- Why repeated retention must use multiplication.

### Part B: Noise Schedule

Run:

```powershell
python code/01_diffusion_from_scratch/01_noise_schedule.py
```

Do not think of this as a difficult implementation task yet. Read it as:

```text
create 1000 pre-chosen noise amounts, then calculate the signal remaining
```

Look at:

- `betas`: per-step noise amounts.
- `alphas`: per-step preserved signal.
- `alpha_bars`: total preserved signal after repeated steps.

### Part C: Corrupt An Image

Run:

```powershell
python code/01_diffusion_from_scratch/02_forward_diffusion_visual.py
```

The image in this exercise is a simple plus sign, so you can focus on the process without downloading MNIST.

Look at:

- `step=0`: completely clean image.
- Early steps: the plus sign remains clear.
- Middle steps: noise competes with the shape.
- Final step: almost only noise remains.

Read the extended teacher notes:

```text
notes/lesson_01_teacher_notes.md
```

Plots are saved in:

```text
outputs/forward_diffusion/
```

## Your Assignment

Do not implement a new model yet. First make sure the noising process is meaningful to you.

1. Run all three scripts.
2. Open the saved plots.
3. Write four short answers in your own words:

```text
1. What does beta_t decide at a single step?
2. Why does alpha_bar multiply alpha values?
3. What does a linear schedule mean?
4. Why is direct sampling of x_t useful for training?
```

When these answers feel natural, you are ready for noise prediction.
