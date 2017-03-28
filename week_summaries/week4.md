# Object Detection
During these two weeks we have tried different CNN architectures to perform object detection. In particular we trained the following networks: YOLO, Tiny YOLO and SSD. As datasets we used again the TT100K dataset, but instead of having crops, we trained with complete images. We have also used the Udacity dataset which has really different training and testing images, in the paper we introduce the solution which leads to a better dataset.

## Code
We have added the following files:
* `models/ssd.py`: this is the Keras implementation of the Single Shot Multibox Detector from [this paper](https://arxiv.org/abs/1512.02325). The original code can be found [here](https://github.com/rykov8/ssd_keras). The author also provides the weights of the network trained on Imagenet, so the implementation allows to either train the model from scratch or load the weights trained on Imagenet.
* `tools/ssd_utils.py`: This file contains many utilities related to the SSD model. It defines how to read the ground truth boxes and convert them to the format used by SSD. It also defines how to map from the output of the model to the final bounding boxes.
* `ssd_eval_detection_fscore.py`: This script computes the precision, recall and F1-score metrics of an SSD model on a given dataset.
* `script analyze dataset`: Computes statistics of a given dataset, like average size of bounding box or number of bounding boxes per class. Can be executed specifying the number of classes -n and dataset name -d. *e.g.  python info.py -n 3 -d Udacity*
* `jupyters/View SSD Boxes.ipynb`: Jupyter notebook useful to debug a SSD model an view the predicted bounding boxes.

## Results
In this section we report the results obtained in the train, validation and test sets of different datasets. We also report the average Frames per second (FPS) that a network can process with a batch size of 128.

For TT100K, the parameter `detection_threshold` was set to 0.3 by evaluating on the validation set.

| TT100K (Train)    | Precision   | Recall  | F1 Score  | FPS |
| ----------------- |:------:| :-----:|:-----:|:-----:|
| YOLO              | 0.834      | 0.777      | 0.805     | 20 |
| Tiny YOLO         | 0.662      | 0.662      | 0.548     | 32 |
| SSD pre-trained   | 0.855      | 0.944      | 0.897     | 97 |
| SSD from scratch  | 0.940      | 0.592      | 0.727     | 97 |

| TT100K (Val)    | Precision   | Recall  | F1 Score  | FPS |
| ----------------- |:------:| :-----:|:-----:|:-----:|
| YOLO              | 0.590      | 0.2      | 0.298     | 20 |
| Tiny YOLO         | 0.5      | 0.061      | 0.109     | 32 |
| SSD pre-trained   | 0.500     | 0.338      | 0.404     | 97 |
| SSD from scratch  | 1      | 0.092      | 0.169     | 97 |

| TT100K (Test)    | Precision   | Recall  | F1 Score  | FPS |
| ----------------- |:------:| :-----:|:-----:|:-----:|
| YOLO              | 0.789      | 0.658      | 0.718     | 20 |
| Tiny YOLO         | 0.581	      | 0.346      | 0.434     | 32 |
| SSD pre-trained   | 0.808      | 0.869      | 0.837    | 97 |
| SSD from scratch  | 0.9      | 0.508      | 0.649     | 97 |

The results on Udacity with different models:

| Udacity (Train)    | Precision   | Recall  | F1 Score  | FPS |
| ----------------- |:------:| :-----:|:-----:|:-----:|
| YOLO              | 0.771      | 0.665      | 0.714     | 20 |
| SSD pre-trained   | 0.087      | 0.763      | 0.157     | 115 |
| SSD w/o Trucks    | 0.572      | 0.757      | 0.652     | 115 |
| SSD from scratch  | 0.030      | 0.707      | 0.058     | 115 |
		
| Udacity (Val)    | Precision   | Recall  | F1 Score  | FPS |
| ----------------- |:------:| :-----:|:-----:|:-----:|
| YOLO              | 0.560      | 0.334      | 0.419     | 20 |
| SSD pre-trained   | 0.075      | 0.533      | 0.132     | 115 |
| SSD w/o Trucks    | 0.518      | 0.540      | 0.529     | 115 |
| SSD from scratch  | 0.026      | 0.547      | 0.051     | 115 |

| Udacity (Test)    | Precision   | Recall  | F1 Score  | FPS |
| ----------------- |:------:| :-----:|:-----:|:-----:|
| YOLO              | 0.523      | 0.258      | 0.346     | 20 |
| SSD pre-trained   | 0.063      | 0.433      | 0.111     | 115 |
| SSD w/o Trucks    | 0.456      | 0.442	    | 0.449     | 115 |
| SSD from scratch  | 0.025      | 0.473      | 0.049     | 115 |

## Instructions
Training an object detector is quite easy. First we have to define a configuration file inside the `config` folder, and then call `train.py` with the desired configuration. To choose between the two detection models implemented, the `model_name` variable should be defined to `"ssd"`or `"yolo"`:
```
python train.py -c config/CONFIG_FILE
```

To evaluate a YOLO model the command is the following:
```
python eval_detection_fscore.py WEIGHTS_FILE DATASET [TINY]
```
Optionally, a third parameter `tiny` can be used if the weights file corresponds to a Tiny-YOLO network.

To evaluate an SSD model the command is the following:
```
python ssd_eval_detection_fscore.py WEIGHTS_FILE DATASET
```

Additionally, the metrics can also be computed along with an option to plot the bounding boxes predicted by the model by using the Jupyter notebook `View SSD Boxes.ipynb` inside the folder `jupyers/`.

## Goals
Level of completeness of the goals of this week
#### Task A
- [x] Analyze TT100K Dataset
- [x] Train YOLO
- [x] Train Tiny-YOLO
- [x] Evaluate metrics

#### Task B
- [x] Read YOLO paper(s)
- [x] Read SSD paper

#### Task C
- [x] Analyze Udacity Dataset
- [x] Train YOLO
- [x] Fxix the problems of the dataset

#### Task D
- [x] Integrate SSD in Keras
- [x] Train on TT100K dataset
- [x] Train on Udacity dataset
- [x] Evaluate SDD on both datasets

#### Task E
- [ ] ...

#### Task F
- [ ] Write report

## Links
[Link](https://docs.google.com/presentation/d/1V-ui0jbUjdvCARN4frC-gQrkKvEKChS92FLr5iQ614o/edit#slide=id.g1d0f8546dc_1_0) to the Google Slide presentation

[Link](https://drive.google.com/open?id=0B3RGXagP6D6sQUptXzhBd3U3Qzg) to a Google Drive with the weights of the model
