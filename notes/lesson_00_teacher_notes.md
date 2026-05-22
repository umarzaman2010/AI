# Lesson 00 Teacher Notes

## What This Lesson Is Really About

Diffusion models look complicated later, but the first building blocks are simple:

- An image is a tensor.
- Noise is a tensor with random values.
- A model is a function with learnable parameters.
- Training means changing those parameters to reduce a loss.

If you understand these four ideas, you have the base needed for DDPM.

## Exercise 1: Tensor And Noise

The clean image is all zeros:

```text
shape = [1, 28, 28]
```

That means:

- `1` channel, because it is grayscale.
- `28` pixels high.
- `28` pixels wide.

`torch.randn_like(clean_image)` creates random Gaussian noise with the same shape.

This matters because diffusion models add noise to images without changing their shape.

## Exercise 2: Gaussian Histogram

When you sample many values from `torch.randn`, most values are near zero.

Fewer values are very large or very small.

That bell-shaped pattern is the standard normal distribution:

```text
mean = 0
standard deviation = 1
```

Diffusion models use this kind of noise during both training and sampling.

## Exercise 3: Tiny Training Loop

This exercise trains a tiny neural network to learn:

```text
y = 2x + 1
```

The model starts with random weight and bias.

After training, it should learn values close to:

```text
weight = 2
bias = 1
```

This is the same training pattern used later for diffusion:

```text
prediction -> loss -> backward -> optimizer step
```

Only the model and data become more complex.

## Before Moving On

You should be able to explain:

- Why the noise tensor has the same shape as the image tensor.
- Why the histogram looks like a bell curve.
- What loss means.
- What `loss.backward()` does.
- What the optimizer changes.

