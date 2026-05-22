# Lesson 05: Sampling

## Goal

Generate images by reversing the diffusion process.

## Concept

Training starts from real images.

Sampling starts from pure noise.

The model repeatedly predicts noise and uses that prediction to move one step toward a cleaner image.

## Sampling Loop

```text
x_T = random Gaussian noise

for t from T down to 1:
    predict noise using model(x_t, t)
    compute x_{t-1}

return x_0
```

## Teacher Explanation

Sampling is like repeatedly asking:

```text
At this noise level, what part of this image looks like noise?
```

Then the sampler removes a carefully scaled amount of that predicted noise.

## Practical Tasks

Task 1:

- Implement the reverse sampling formula.
- Generate one image from random noise.

Task 2:

- Save intermediate denoising steps.
- Make a grid showing noise gradually becoming an image.

Task 3:

- Compare samples from early and later checkpoints.

## Checkpoint Questions

- Why does sampling start from noise?
- Why does sampling run backward through timesteps?
- Why is sampling slower than a normal neural network prediction?
- What happens if the model predicts bad noise?

