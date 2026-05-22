# Lesson 07: Latent Diffusion

## Goal

Understand why modern image diffusion systems often operate in latent space.

## Concept

Pixel-space diffusion works directly on images.

Latent diffusion first compresses an image using an autoencoder, then runs diffusion in the smaller latent representation.

This is faster because the model denoises a smaller tensor.

## Stable Diffusion Components

Stable Diffusion is usually explained as four main parts:

- Text encoder: converts prompt text into embeddings.
- U-Net: denoises latent tensors.
- Scheduler: controls the denoising timesteps.
- VAE: decodes the final latent into an image.

## Why It Matters

Your MNIST DDPM teaches the core idea.

Stable Diffusion scales it by:

- Using latent space.
- Conditioning on text.
- Training on much larger data.
- Using stronger U-Nets or transformer-based denoisers in newer systems.

## Practical Tasks

Task 1:

- Draw the Stable Diffusion pipeline in your own words.

Task 2:

- Read Jay Alammar's visual guide.
- Write a summary in `notes/stable_diffusion_summary.md`.

Task 3:

- Use Hugging Face Diffusers only after completing your from-scratch DDPM.

## Checkpoint Questions

- What is the difference between pixel space and latent space?
- Why is latent diffusion faster?
- What does the VAE do?
- What does the text encoder do?
- What does the scheduler do?

