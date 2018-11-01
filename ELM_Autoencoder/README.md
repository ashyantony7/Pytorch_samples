## Extreme Learning Machine as AutoEncoder

### Algorithm for ELM

Step 1: Randomly assign input weights and biases for linear connected layer.

Step 2: Calculate the hidden layer output matrix H.

Step 3: Calculate the output weights Beta from the Targets T and the Hidden layer H.
For autoencoder the output is same as the input.

### Results

The algorithm is tested with a Single Layer Feed Forward Network (SLFN) with 10 hidden units and the
results are displayed below

![alt text](https://raw.githubusercontent.com/ashyantony7/Pytorch_Samples/master/images/ELMAE.png)
(i) First row - Original Images (ii) Second row - Decoded images

### References

1. [Guang-Bin Huang , Qin-Yu Zhu, Chee-Kheong Siew, "Extreme learning machine: Theory and applications"](https://doi.org/10.1016/j.neucom.2005.12.126)
2. [Kai Sun, Jiangshe Zhang, Chunxia Zhang, Junying Hu, "Generalized extreme learning machine autoencoder and a new deep neural network"](https://doi.org/10.1016/j.neucom.2016.12.027)


