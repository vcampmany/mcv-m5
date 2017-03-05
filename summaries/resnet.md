#Deep Residual Learning for Image Recognition

As we see on the last state of the art of neural network models, depth is crucial in order to improve performance. Anyway, as deeper the network as bigest degradation problem we can found.

The authors of this [paper][[1]](https://arxiv.org/abs/1512.03385) expose that adding identity mapping layers would give us, at least, the same results as with the original models. So they redefine the original mapping, instead of computing the whole transformation from x through conv-relu-conv series to H(x), they compute F(x),the term to add to the input x. This residual mapping is easier to optimize than the original inreferenced mapping, also, on backpropagation, the gradient will flow easily because we have addition operations, which distributes the gradient.

Their main contribution is the definition on this architecture which contains residual building blocks (ResBlocks). ResBlocks are simple layer connected with rectified linear units (ReLU) and a pass-through below that feeds through the information from previous layers unchanged. They also test different modifications of these blocks as bottleneck blocks, with three layers where the middle one restricts the flow of information using fewer inputs and outputs. Another modification is to test different types of pitch connections including a full projection matrix.

The rest of the work tests the performance of the network. And they demonstrate that they can train deeper nets by improving performance. Finally, created a ResNet of 152 layers.

### References
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2015). Deep Residual Learning for Image Recognition.
