# Lesson 06: Conditioning

## Goal

Teach the model to generate specific kinds of images.

## Concept

Unconditional diffusion learns:

```text
Generate something from the training distribution.
```

Conditional diffusion learns:

```text
Generate something from the training distribution that matches condition y.
```

For MNIST, `y` can be a digit label.

## Class Conditioning

Add a class embedding:

```text
class_id -> embedding vector
```

Combine this with the timestep embedding so the U-Net knows both:

- How noisy the image is.
- Which class it should generate.

## Classifier-Free Guidance

During training, sometimes remove the condition.

At sampling time, compare:

- unconditional prediction
- conditional prediction

Then push the result more strongly toward the condition.

## Practical Tasks

Create code in `code/04_conditioning/`.

Task 1:

- Add label embeddings.
- Train a class-conditioned MNIST DDPM.

Task 2:

- Generate digits 0 through 9.
- Save a grid.

Task 3:

- Add classifier-free guidance.
- Try guidance scales like 0, 1, 2, 4.

## Checkpoint Questions

- What is the condition in class-conditioned MNIST?
- Where can label information enter the model?
- What does classifier-free guidance change during sampling?
- What happens if guidance is too strong?

