# Lesson 00B: Mean, Spread, Variance, And Scaling

## Why You Need This Before Diffusion

In Lesson 01 we mix an image with random noise. To understand the mixing formula, you need only one probability idea:

```text
How strong or spread out is the random noise?
```

Do this lesson before worrying about `beta`, `alpha`, or square roots.

## Start With Random Noise Values

Imagine we sample five random noise values:

```text
[-0.2, 0.1, -1.3, 0.8, 0.0]
```

Noise values are not all the same. Some are near zero; some move farther away.

For an image, each pixel gets one such random value. Stronger noise means pixel values can be pushed farther from their original values.

## Mean: The Center

The **mean** is simply the average.

```text
mean = add all values, then divide by how many values there are
```

Gaussian noise used in diffusion is centered around zero:

```text
positive and negative changes roughly balance out
```

For many samples from `torch.randn`, the mean should be close to:

```text
0
```

## Spread: How Far Values Wander From The Center

Compare two noise groups:

```text
small spread: [-0.2, 0.1, 0.0, -0.1, 0.2]
large spread: [-2.0, 1.0, 0.0, -1.0, 2.0]
```

Both are centered near zero, but the second group is stronger noise because its values wander farther from zero.

## Standard Deviation: A Number For Spread

**Standard deviation** is a number describing typical distance from the mean.

You can first think of it this way:

```text
standard deviation small -> weak, tightly grouped noise
standard deviation large -> strong, widely spread noise
```

Standard Gaussian noise from `torch.randn` has approximately:

```text
mean = 0
standard deviation = 1
```

## Variance: Spread Written In Squared Units

**Variance** is closely related to standard deviation:

```text
variance = standard deviation squared
standard deviation = square root of variance
```

Examples:

| Standard deviation | Variance |
| --- | --- |
| `1` | `1` |
| `2` | `4` |
| `0.5` | `0.25` |

Why use variance? Probability formulas often become cleaner when spread is expressed as variance. Diffusion papers describe how much random uncertainty is added using variance.

## What Does Amplitude Mean Here?

You do not need the word **amplitude**. In this course, replace it in your mind with:

```text
multiplier applied to values
```

For example:

```text
noise = torch.randn(...)
scaled_noise = 2 * noise
```

Multiplying by `2` makes the random values twice as large in magnitude. Their standard deviation becomes approximately `2` instead of `1`.

## The Important Scaling Rule

If you multiply random values by `c`:

```text
new standard deviation = abs(c) * old standard deviation
new variance = c^2 * old variance
```

For standard Gaussian noise, the old variance is `1`.

If we want the new noise variance to be `0.04`, what multiplier should we use?

```text
c^2 = 0.04
c = sqrt(0.04) = 0.2
```

So:

```text
sqrt(0.04) * epsilon
```

has variance `0.04` when `epsilon` has variance `1`.

This is the only reason square roots appear in the diffusion noising equation.

## Connect This To Diffusion

In a diffusion step, we choose:

```text
beta_t = amount of noise variance added at this step
```

Since `epsilon` already has variance `1`, the noise term becomes:

```text
sqrt(beta_t) * epsilon
```

because:

```text
variance of sqrt(beta_t) * epsilon = beta_t
```

The old image is also scaled to keep the mixture controlled:

```text
sqrt(1 - beta_t) * previous_image
```

Later we name:

```text
alpha_t = 1 - beta_t
```

so the one-step formula becomes:

```text
x_t = sqrt(alpha_t) * x_(t-1) + sqrt(beta_t) * epsilon_t
```

## Practical Experiment

Run:

```powershell
python code/00_math_basics/04_spread_and_scaling.py
```

It will generate standard Gaussian noise and multiply it by different values.

You should notice:

- Multiplying noise by `0.2` gives standard deviation near `0.2` and variance near `0.04`.
- Multiplying noise by `2.0` gives standard deviation near `2.0` and variance near `4.0`.
- This is why a desired variance uses a square-root multiplier.

## Checkpoint Questions

Answer these before returning to Lesson 01:

```text
1. What does mean tell us?
2. What does standard deviation tell us?
3. How are variance and standard deviation related?
4. If noise variance should be 0.09, why do we multiply standard noise by 0.3?
```

