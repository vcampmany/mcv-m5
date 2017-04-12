# Semantic Segmentation
During these two weeks we have tried different CNN architectures to perform semantic segmentation. In particular we trained the following networks: FCN8, and Segnet. As datasets we used Camvid, Cityscapes and Kitti.

## Code
We have added the following files:
* `models/segnet.py`:  this is the Keras implementation of the SegNet from [this paper](https://arxiv.org/pdf/1511.00561.pdf). It has both the basic and VGG implementations of the model.
* `jupyters/Outputs SegNet`: Jupyter notebook useful to debug a SegNet model an plot the semantic segmentation predicted. It loads a trained SegNet and predicts segmentations from images, showing the segmentation in the desired color map. 
* `script analyze dataset`: Extracts probability distribution for each dataset class, splited in training, test and validation sets. Allows to analyze our datasets distribution in order to further understanding of our results. Can be executed specifying the number of classes -n and dataset name -d. e.g. python info.py -n 12 -d camvid
## Results
In this section we report the results obtained in the train and validation sets of different datasets.


| FCN8 (Train)    | Accuracy   | jaccard  | Loss  | 
| ----------------- |:------:| :-----:|:-----:|
| Camvid            | 0.97      | 0.85      | 0.074     | 
| Cityscapes        | 0.936      | 0.627      | 0.194     |
| Kitti             | 0.961      | 0.752      | 0.101     |


| FCN8 (Val)    | Accuracy   | jaccard  | Loss  | 
| ----------------- |:------:| :-----:|:-----:|
| Camvid            | 0.926      | 0.645      | 0.287     | 
| Cityscapes        | 0.901      | 0.493      | 0.347     |
| Kitti             | 0.794      | 0.401      | 1.309     |



For Segnet: 

| Segnet (Train)    | Accuracy   | jaccard  | Loss  | 
| ----------------- |:------:| :-----:|:-----:|
| Camvid (Basic)    | 0.949      | 0.746      | 0.135     | 
| Camvid (VGG)      | 0.98      | 0.913      | 0.046     |
| Camvid (VGG+Dataug) | 0.971 | 0.865	| 0.068 |



| Segnet (Val)    | Accuracy   | jaccard  | Loss  | 
| ----------------- |:------:| :-----:|:-----:|
| Camvid (Basic)    | 0.918      | 0.626      | 0.264     | 
| Camvid (VGG)      | 0.946      | 0.737      | 0.274     |
| Camvid (VGG+Dataug) | 0.951    | 0.865	    | 0.199     |


## Instructions
Training these networks for semantic segmentation is quite easy. First we have to define a configuration file inside the `config` folder, and then call `train.py` with the desired configuration. To choose between the two detection models implemented, the `model_name` variable should be defined to `"fcn8"` `"segnet_basic"` or `"segnet_vgg"`:
```
python train.py -c config/CONFIG_FILE
```


Additionally, we can plot the semantic segmentation predicted by the SegNet model by using the Jupyter notebook `Outputs SegNet.ipynb` inside the folder `jupyters/`.

## Goals
Level of completeness of the goals of this week
#### Task A
- [x] Analyze the dataset
- [x] Train FCN8
- [x] Evaluate metrics

#### Task B
- [x] Read FCN paper(s)
- [x] Read SegNet paper

#### Task C
- [x] Train the network on a different dataset

#### Task D
- [x] Integrate SegNetVGG and SegNet Basic
- [x] Evaluate SegNetVGG and SegNet Basic on CamVid Dataset

#### Task E
- [x] Try with Data Augmentation
- [x] Tool to visualize Segmentation

#### Task F
- [x] Write report

## Links
[Link](https://docs.google.com/presentation/d/1V-ui0jbUjdvCARN4frC-gQrkKvEKChS92FLr5iQ614o/edit#slide=id.g1d0f8546dc_1_0) to the Google Slide presentation

[Link](https://drive.google.com/open?id=0B3RGXagP6D6sQ3ZwUDFkdW84N00) to a Google Drive with the weights of the model
