# Object Detection
During these two weeks we have tried different CNN architectures to perform object detection. In particular we trained the following networks: YOLO, Tiny YOLO and SSD. As datasets we used again the TT100K dataset, but instead of having crops, we trained with complete images. We have also used the Udacity dataset which has really different training and testing images, in the paper we introduce the solution which leads to a better dataset.

## Code
We have added the following files:
* To be completed

## Results
In this section we report the results obtained in the train, validation and test sets of different datasets.

| TT100K (Train)    | Precision   | Recall  | F1 Score  |
| ----------------- |:------:| :-----:|:-----:|
| YOLO              | 0.834      | 0.777      | 0.805     |
| Tiny YOLO         | 0.662      | 0.662      | 0.548     |
| SSD pre-trained   | 0.959      | 0.743      | 0.837     |
| SSD from scratch  | 0.940      | 0.592      | 0.727     |

| TT100K (Val)    | Precision   | Recall  | F1 Score  |
| ----------------- |:------:| :-----:|:-----:|
| YOLO              | 0.590      | 0.2      | 0.298     |
| Tiny YOLO         | 0.5      | 0.061      | 0.109     |
| SSD pre-trained   | 0.909      | 0.153      | 0.263     |
| SSD from scratch  | 1      | 0.092      | 0.169     |

| TT100K (Test)    | Precision   | Recall  | F1 Score  |
| ----------------- |:------:| :-----:|:-----:|
| YOLO              | 0.789      | 0.658      | 0.718     |
| Tiny YOLO         | X      | X      | X     |
| SSD pre-trained   | 0.918      | 0.653      | 0.763     |
| SSD from scratch  | 0.9      | 0.508      | 0.649     |

The results on Udacity with different models:

| Udacity (Train)    | Precision   | Recall  | F1 Score  |
| ----------------- |:------:| :-----:|:-----:|
| YOLO              | 0.771      | 0.665      | 0.714     |
| Tiny YOLO         | X      | X      | X     |
| SSD pre-trained   | 0.087      | 0.763      | 0.157     |
| SSD from scratch  | 0.030      | 0.707      | 0.058     |

| Udacity (Val)    | Precision   | Recall  | F1 Score  |
| ----------------- |:------:| :-----:|:-----:|
| YOLO              | 0.560      | 0.334      | 0.419     |
| Tiny YOLO         | X      | X      | X     |
| SSD pre-trained   | 0.075      | 0.533      | 0.132     |
| SSD from scratch  | 0.026      | 0.547      | 0.051     |

| Udacity (Test)    | Precision   | Recall  | F1 Score  |
| ----------------- |:------:| :-----:|:-----:|
| YOLO              | 0.523      | 0.258      | 0.346     |
| Tiny YOLO         | X      | X      | X     |
| SSD pre-trained   | 0.063      | 0.433      | 0.111     |
| SSD from scratch  | 0.025      | 0.473      | 0.049     |

## Instructions
Explain how to run (train and eval)

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
