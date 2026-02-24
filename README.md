# FashionMNIST Generative Modeling Project

This project explores latent-variable generative models on the
FashionMNIST dataset.\
The goal is to build, improve, and evaluate VAE-based models in a
structured way.

------------------------------------------------------------------------

## 1. Data Preparation & EDA

We load the FashionMNIST dataset and split it into train/validation/test
sets.

In this section we: - Visualize sample images with labels - Check class
distribution - Inspect pixel intensity histograms

This ensures the dataset is balanced and confirms the input scale
[0, 1], which is important for BCE-based VAE training.

------------------------------------------------------------------------

## 2. Baseline VAE (MLP)

We first implement a fully-connected Variational Autoencoder.

Structure: - Encoder -> outputs $\mu$ and log($\sigma^2$) - Reparameterization
trick - Decoder -> reconstructs 28×28 images - Loss = Reconstruction
(BCE) + KL divergence

We evaluate: - Reconstruction quality - Prior sampling quality - Test
loss components (reconstruction + KL)

This provides a reference model for later improvements.

------------------------------------------------------------------------

## 3. Improved VAE (Convolutional + KL Warmup)

To improve generation quality, we replace the MLP with a convolutional
encoder/decoder.

Enhancements: - Conv layers for better spatial modeling - KL warmup for
more stable training - Larger latent dimension

We compare the improved model against the baseline using: -
Reconstructions - Prior samples - Test loss metrics

This model generally produces sharper and more realistic samples.

------------------------------------------------------------------------

## 4. beta-VAE Analysis

We train models with different beta values to study disentanglement.

Lower beta: - Better reconstructions - Weaker latent regularization

Higher beta: - Stronger latent structure - Slightly worse reconstruction
quality

We also perform latent traversal: - Vary one latent dimension at a
time - Observe how image attributes change

This helps interpret what the latent space has learned.

------------------------------------------------------------------------

## 5. Conditional VAE (CVAE)

We extend the model by conditioning on class labels.

Changes: - One-hot labels are concatenated to encoder input - Labels are
also injected into the decoder

This allows class-controlled generation.

We evaluate: - Reconstruction loss - Class controllability (using a
pretrained FashionResNet18 classifier) - Feature-based FID proxy score

Generated samples are normalized before being passed to the classifier
to match its training distribution.

------------------------------------------------------------------------

## 6. Evaluation Protocol

-   VAE/CVAE are trained on raw images in [0, 1]
-   Classifier uses its saved mean/std normalization
-   Generated samples are normalized only for classifier evaluation

This avoids data leakage and ensures fair comparison.

------------------------------------------------------------------------

## Summary

The project demonstrates: - Implementation of VAEs from scratch -
Architectural improvements using convolutions - Latent space analysis
with beta-VAE - Label-controlled generation with CVAE - Quantitative
evaluation using a pretrained classifier

The final comparison table reports reconstruction loss, KL divergence,
total loss, and FID proxy where applicable.
