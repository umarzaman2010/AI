# Lesson 01 Teacher Notes: Why The Formula Exists

## First: What Problem Are We Setting Up?

We want a neural network that can learn to remove noise from images.

To train such a network, we need examples of:

```text
image before corruption -> image after controlled corruption
```

So we create a forward process ourselves. It takes a real image and destroys information slowly with random Gaussian noise.

Later, the neural network will try to reverse this destruction.

## Think In Terms Of Signal And Noise

An image contains **signal**: structure we care about, such as strokes of a digit or edges of an object.

Gaussian noise adds random variation that hides that structure.

At every diffusion step we choose to keep most of the previous signal and inject a small amount of new noise:

```text
x_t = sqrt(alpha_t) * x_(t-1) + sqrt(beta_t) * epsilon_t
```

This equation is not magic. It says:

```text
new image = scaled old image + scaled random noise
```

## Why `beta`?

`beta_t` is a design choice. It controls how strongly step `t` corrupts the image.

If `beta_t` is tiny:

- Very little new noise is introduced.
- The image changes gently.

If `beta_t` is large:

- More new noise is introduced.
- The image is damaged more strongly.

DDPM uses many small beta values because it is easier to learn to undo small, gradual corruption than one violent corruption step.

## Why `alpha = 1 - beta`?

`beta` describes the part assigned to noise variance at a step.

We allocate the rest to the existing signal:

```text
alpha_t + beta_t = 1
alpha_t = 1 - beta_t
```

If:

```text
beta_t = 0.02
```

then:

```text
alpha_t = 0.98
```

That step assigns most of the variance to the previous image and a small portion to newly added noise.

## Why `sqrt(alpha)` And `sqrt(beta)`?

This is an important subtlety.

`alpha` and `beta` describe variance. But in the equation we multiply image values and noise values by amplitudes.

If an amplitude is multiplied by `c`, its variance is multiplied by `c^2`.

So to contribute variance `alpha`, the image is multiplied by `sqrt(alpha)`. To contribute variance `beta`, noise is multiplied by `sqrt(beta)`.

You do not need to prove this yet, but you should know why the square roots appear.

## Why Does `alpha_bar` Multiply?

Let us ignore the newly added noise for a moment and track only how much original signal survives.

After the first step:

```text
original signal retained = alpha_1
```

After the second step, only `alpha_2` of the already-retained signal survives:

```text
original signal retained = alpha_1 * alpha_2
```

After `t` steps:

```text
original signal retained = alpha_1 * alpha_2 * ... * alpha_t
```

We give that product a name:

```text
alpha_bar_t = product of all alpha values from step 1 to t
```

The clean image amplitude in the final direct formula becomes `sqrt(alpha_bar_t)` because, again, `alpha_bar_t` tracks a variance amount.

## A Numerical Example

Use unrealistically large numbers to make the idea visible:

```text
alpha_1 = 0.90
alpha_2 = 0.80
alpha_3 = 0.70
```

Then:

```text
alpha_bar_1 = 0.90
alpha_bar_2 = 0.90 * 0.80 = 0.72
alpha_bar_3 = 0.90 * 0.80 * 0.70 = 0.504
```

After three steps, only about half of the original signal variance remains.

## What A Schedule Really Is

A schedule is not a neural network and not a training method.

It is just a table decided in advance:

```text
step 1 -> add this much noise
step 2 -> add this much noise
...
step T -> add this much noise
```

A linear beta schedule means the values form a straight increasing line from a chosen start to a chosen end.

In code:

```python
betas = torch.linspace(0.0001, 0.02, 1000)
```

This creates 1000 beta values, increasing evenly.

## Why We Can Jump Directly To `x_t`

If training required adding noise one step at a time, creating a training example at step 700 would take 700 operations.

Fortunately, repeated Gaussian corruption can be combined into:

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
```

Now we can:

- Pick a clean image.
- Pick any random timestep.
- Add the correct total noise immediately.
- Ask the network to predict the noise.

This shortcut is one reason DDPM training is practical.

## A Correction To The Earlier Visual

The clean image should be called `step=0`.

The first beta value produces `step=1`, the first slightly noisy image.

The updated visual script now labels the clean starting point separately so the timeline is easier to understand.

## You Are Ready For The Next Lesson When

You can explain, without memorizing:

- A noising step mixes retained image signal with new Gaussian noise.
- `beta_t` chooses noise amount at one step.
- `alpha_t = 1 - beta_t` is signal retained at one step.
- `alpha_bar_t` is multiplication because repeated retention compounds.
- A schedule is a pre-chosen list of beta values.
- The direct formula quickly makes a noisy training example at any timestep.

