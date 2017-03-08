# Rethinking the Inception Architecture for Computer Vision

In this paper [(link)](https://arxiv.org/abs/1512.00567) the authors revisit the Inception architecture presented in [[1]](https://arxiv.org/abs/1409.4842) and propose some general design principles and optimization ideas to improve the architecture.
The main design principles are:
* __Avoid representational bottlenecks__: The size of the activations of each layer should smoothly decrease from the input to the output, without abrupt changes of size.
* __Balance width and depth__: to increase the capacity of the model it is better to increase both width (number of parameters per layer) and depth (number of layers) in parallel.

The authors show how convolutions with large filters can be factorized into stacked convolutions of smaller filters retaining the same performance. For example, a 5x5 convolution can be factorized into two 3x3 convolutions, reducing the computations by 28%.
In order to perform a non-aggresive size reduction to avoid a representational bottleneck, a new inception module is proposed. It consists on using 2 blocks in parallel (a standard pooling and a convolution sith stride 2) and then concatenating their outputs. It is computationally more efficient than the standard convolution+pooling.

With all the tricks described in the paper the authors create the architecture called "Inception-v3", which obtains a 21.2% top-1 error rate for a single-crop in Imagenet, much better than the 29% obtained by GoogleNet.

### References
[1] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). Going deeper with convolutions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 1-9).
