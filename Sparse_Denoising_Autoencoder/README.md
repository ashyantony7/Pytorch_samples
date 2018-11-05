
## Sparse Denoising Stacked Autoencoder

## Algorithm

1. Sparse auto-encoder is a feed-forward neural network with unsupervised learning using back propagation and batch gradient descent algorithm. 
2. The loss function is modified with the KL divergence and sparsity parameter
3. The autoencoder is denoising as the input images are added with noise and the target image is without noise

### Results

The algorithm is trained with 1000 epochs and with embedeed length of 14 and the results are displayed below

![alt text](https://raw.githubusercontent.com/ashyantony7/Pytorch_Samples/master/images/SDSAE.png)

(i) First row - Original Images (ii) Second row - Decoded images

### References

1. [Research of stacked denoising sparse autoencoder](https://link.springer.com/article/10.1007/s00521-016-2790-x)
2. [Anush Sankaran, Prateekshit Pandey, Mayank Vatsa, Richa Singh, "On Latent Fingerprint Minutiae Extraction using Stacked Denoising Sparse AutoEncoders"](https://doi.org/10.1109/BTAS.2014.6996300)
