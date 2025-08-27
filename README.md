# Removing GAN Model Fingerprints
In this project we try to remove or supress model fingerprints from GANs with selected filters. Francescco Marra, et al, with their paper "Do GANs leave artificial fingerprints?" has been able to show that source attribution can be done on GAN generated images by simply correlating unique noise patterns (PRNU) to the a reference pattern aggregated by the taking the mean on all the synthetic images here: https://doi.org/10.48550/arXiv.1812.11842 .

So we would try to remove them by subtracting the reference pattern aggregated from a collection of images from a particular GAN from the images generated from the same GAN to see if this would be effective in removing or supressing the model fingerprints left by these GANs.

## Method Overview