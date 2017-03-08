# Very Deep Convolutional Networks for Large-Scale Image Recognition

The main contribution of this paper is the use of smaller convolutions (3x3) and consequently a deeper architecture. By stacking small size convolutions we obtain the same receptive field as using one bigger convolution. However, with this scheme we use less parameters and more non-linearities.

Several architectures were introduced with a number of layers ranging from 11 to 19. All of them used 3x3 convolutions in most of the layers. Some architectures also introduced 1x1 convolutions which introduces an additional non-linearity without changing the receptive field. In all the architectures, the convolutional layers are followed by 3 fully connected layers and the softmax output.

After each maxpooling layer, they double the amount of used filters. The fact gives strength to the idea of reducing the spatial size of the input tensor, but increasing the depth as we go deeper into the network.

Regarding the training they use scale jittering as a way of doing data augmentation [[1]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). The reason behind this idea is that the objects within an image can be of different size.

The obtained results in the ILSVRC outperform the previous winners in ILSVRC-2012 and ILSVRC-2013 both in the top-1 and top-5 metric.

### References
[1] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).
