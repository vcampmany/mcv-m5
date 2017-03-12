# Object Recognition
For the stage of Object Recognition we have tested different CNN architectures on different datasets. More precisely, we have tested the VGG16 model, the Inception V3, a ResNet of 50 layers and a Wide Residual Network, presented [in this paper] (https://arxiv.org/pdf/1605.07146.pdf). The datasets used to evaluate these CNN architectures are the [TT100K dataset](http://cg.cs.tsinghua.edu.cn/traffic-sign/), the [BelgiumTS dataset](http://btsd.ethz.ch/shareddata/) and the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php). The first two are for traffic sign recognition, whereas KITTI is a more generic object recognition dataset and it has classes such as car, pedestrian or truck.

## Code
We have added the following files:
* `models/wideresnet.py`: this is the Keras implementation of the Wide Residual Network used for Imagenet presented [here](https://arxiv.org/pdf/1605.07146.pdf). The original code is from Torch and can be found [here](https://github.com/szagoruyko/wide-residual-networks/tree/master/pretrained). The author also provides the weights of the network trained on Imagenet, so the implementation allows to either train the model from scratch or load the weights trained on Imagenet.
* `models/inceptionV3.py`: explain
* `models/resnet.py`: Keras implementaion of the [ResNet](https://arxiv.org/pdf/1512.03385.pdf). As the model is already implemented in keras we just needed to integrate the model to our framework.
* `models/keras_vgg16_l2reg.py`: We have adapted the Keras model of the VGG16 network to accept l2 regularization (weight decay). The call has the same format as the Keras one, but we have added an extra argument to the function call the l2 regularization trade off.
* `script analyze dataset`: explain

## Results
In this section we report the results obtained in the test set of different datasets.

| TT100K            | Loss   | Accuracy  |
| ----------------- |:------:| :-----:|
| VGG baseline      | 0.6443 | 0.9561 |
| VGG resize to 112 | 0.6082 | 0.9531 |
| VGG crop 224      | 0.3352 | 0.9690 |
| VGG normalize featurewise | 0.3633 | 0.9393 |
| VGG normalize Imagenet    | 0.5351 | 0.9609 |
| VGG l2_reg=0.0005         | 0.4517 | 0.9524 |
| InceptionV3 scratch       | 1.5654 | 0.4118 |
| InceptionV3 finetunning   | 1.0571 | 0.6997 |
| Resnet scratch       | 0.2235 | 0.9706 |
| Resnet finetunning   | 0.5185 | 0.8597 |
| Wide RN scratch      | 0.1635 | 0.9751 |
| Wide RN finetunning  | 0.1316 | 0.9821 |
| Wide RN dataug  | x | x |

The Wide Resnet used in the experiments is the WRN-50-2-bottleneck model used by the author on the Imagenet dataset, which achieves an error rare 3% lower than the original Resnet of 50 layers.

| KITTI             | Loss   | Accuracy  |
| ----------------- |:------:| :-----:|
| VGG from scratch  | x | x |
| VGG finetunned    | x | x |

For the tests on the KITTI dataset we had to reduce the learning rate from 0.0001 to 0.00001 (using RMSProp optimizer). Otherwise, at the middle of the training stage the loss would suddenly increase.

| BTS            | Loss   | Accuracy  |
| ----------------- |:------:| :-----:|
| VGG transfer      | 0.4755 | 0.9638 |

## Instructions
The usage of the code is easy. First we have to define a configuration file inside the `config` folder, and then call `train.py` with the desired configuration:
```
python train.py -c config/CONFIG_FILE
```

For example, to run the VGG16 on the KITTI dataset, training from scratch the command is
```
python train.py -c config/kitti_scratch.py -e kitti_scratch
```
Where `config/kitti_scratch.py` is the configuration file that has the details of the training procedure.

## Goals
Level of completeness of the goals of this week
#### Task A
- [x] Analyze Dataset
- [x] Run and evaluate VGG
- [x] Evaluate different techniques
  - [x] Resize vs Crop
  - [x] Different pre-processings
- [x] Transfer Learning to BelgiumTS dataset

#### Task B
- [x] VGG from scratch on KITTI dataset
- [x] VGG finetunned on KITTI dataset

#### Task C
- [x] Implement InceptionV3 in Keras
- [x] Train from scratch on TT100K dataset
- [x] Train finetunned on TT100K dataset

#### Task D
- [x] Implement Resnet in Keras
- [x] Train from scratch on TT100K dataset
- [x] Train finetunned on TT100K dataset

#### Task E
- [x] Implement VGG that accepts l2 regularization
- [ ] Train with data augmentation
- [x] Different Learning rate (with KITTI)
- [x] Implement Another architechture (Wide Resnet)

#### Task F
- [ ] Write report

## Links
Link to the Google Slide presentation
Link to a Google Drive with the weights of the model
