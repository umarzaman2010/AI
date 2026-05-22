# Lesson 03: U-Net

## Goal

Build the denoising network used by many diffusion models.

## Concept

A U-Net has:

- Downsampling path: captures larger context.
- Bottleneck: compressed representation.
- Upsampling path: reconstructs spatial detail.
- Skip connections: preserve fine details.
- Timestep embedding: tells the model how noisy the input is.

## Why U-Net?

Diffusion needs both:

- Local detail: edges, strokes, texture.
- Global structure: digit identity, object shape.

The U-Net handles both because it processes multiple resolutions.

## Practical Tasks

Create code in `code/02_unet/`.

Task 1:

- Build a simple convolution block.
- Build downsample and upsample blocks.
- Add skip connections.

Task 2:

- Add sinusoidal timestep embeddings.
- Inject timestep embeddings into the network.

Task 3:

- Verify:

```text
input shape:  [B, C, H, W]
output shape: [B, C, H, W]
```

## Checkpoint Questions

- Why does diffusion use a U-Net instead of a plain MLP?
- What do skip connections preserve?
- Why is the output shape the same as the input shape?
- What information does the timestep embedding provide?

