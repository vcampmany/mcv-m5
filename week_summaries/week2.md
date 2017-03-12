# Object Recognition
For the stage of Object Recognition we have tested different CNN architectures on different datasets. More precisely, we have tested the VGG16 model, the Inception V3, a ResNet of 50 layers and a Wide Residual Network, presented [in this paper] (https://arxiv.org/pdf/1605.07146.pdf). The datasets used to evaluate these CNN architectures are the [TT100K dataset](http://cg.cs.tsinghua.edu.cn/traffic-sign/), the [BelgiumTS dataset](http://btsd.ethz.ch/shareddata/) and the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php). The first two are for traffic sign recognition, whereas KITTI is a more generic object recognition dataset and it has classes such as car, pedestrian or truck.

## Code
We have added the following files:
* `models/wideresnet.py`: this is the Keras implementation of the Wide Residual Network used for Imagenet presented [here](https://arxiv.org/pdf/1605.07146.pdf). The original code is from Torch and can be found [here](https://github.com/szagoruyko/wide-residual-networks/tree/master/pretrained). The author also provides the weights of the network trained on Imagenet, so the implementation allows to either train the model from scratch or load the weights trained on Imagenet.
* `models/inceptionV3.py`: explain
* `models/resnet.py`: explain
* `models/keras_vgg16_l2reg.py`: explain
* `script analyze dataset`: explain

## Results
Results of the different experiments

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
