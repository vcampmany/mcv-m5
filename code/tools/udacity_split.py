import os
import glob
import shutil

TRAIN_PATH = "/home/master/Datasets/detection/Udacity/train"
TEST_PATH = "/home/master/Datasets/detection/Udacity/test"
VAL_PATH = "/home/master/Datasets/detection/Udacity/valid"

DEST_TRAIN_PATH = "/home/master/experiments/Datasets/detection/Udacity/train"
DEST_TEST_PATH = "/home/master/experiments/Datasets/detection/Udacity/test"
DEST_VAL_PATH = "/home/master/experiments/Datasets/detection/Udacity/valid"

list1 = glob.glob(TRAIN_PATH + "/*.jpg")
list1.sort()

list2 = glob.glob(TEST_PATH + "/*.jpg")
list2 += glob.glob(VAL_PATH + "/*.jpg")
list2.sort()

train1 = int(round(len(list1) * 0.6))
train2 = int(round(len(list2) * 0.6))
trainList = list1[0:train1] + list2[0:train2]

val1 = int(round(len(list1) * 0.8))
val2 = int(round(len(list2) * 0.8))
valList = list1[train1:val1] + list2[train2:val2]

testList = list1[val1:] + list2[val2:]

for impath in trainList:
	txt = impath.replace(".jpg", ".txt")
	imfilename = impath.split('/')[-1]
	txtfilename = txt.split('/')[-1]
	shutil.copyfile(impath, DEST_TRAIN_PATH+'/'+imfilename)
	shutil.copyfile(txt, DEST_TRAIN_PATH+'/'+txtfilename)
	
	
for impath in testList:
	txt = impath.replace(".jpg", ".txt")
	imfilename = impath.split('/')[-1]
	txtfilename = txt.split('/')[-1]
	shutil.copyfile(impath, DEST_VAL_PATH+'/'+imfilename)
	shutil.copyfile(txt, DEST_VAL_PATH+'/'+txtfilename)
	
for impath in valList:
	txt = impath.replace(".jpg", ".txt")
	imfilename = impath.split('/')[-1]
	txtfilename = txt.split('/')[-1]
	shutil.copyfile(impath, DEST_TEST_PATH+'/'+imfilename)
	shutil.copyfile(txt, DEST_TEST_PATH+'/'+txtfilename)

	

