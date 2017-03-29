from os import listdir
from os.path import isfile, join
import numpy as np
import os as os
from PIL import Image
import argparse

sets = ['train', 'test', 'valid']
dataset = 'Udacity'
nclases = 3

parser = argparse.ArgumentParser(description='Analize data')
parser.add_argument('-n', '--number_classes', type=int, default=None, help='Number of classes')
parser.add_argument('-d', '--dataset_name', type=str, default=None, help='Dataset name')
arguments = parser.parse_args()

dataset = arguments.dataset_name
nclases = arguments.number_classes

for s in sets:
    print("")
    path = '../../../Datasets/detection/'+dataset+'/'+s+'/'
    #info_file = open(datset+'/'+s+'_info/distr.txt', 'w')

    info_file = open(dataset+'_own'+'/'+s+'_info/distr.txt', 'w')
    cont = []
    for i in range(nclases):
        cont.append([])
        cont[i].append(0) #num bboxes per class
        cont[i].append(0) #average size per calss
        cont[i].append(0) #averge ratio per class
        cont[i].append(float('inf')) # min size per class
        cont[i].append(0) #max size per clas
        cont[i].append(float('inf')) # min ratio per class
        cont[i].append(0) #max ratio per clas



    for file in os.listdir(path):
        if file.endswith(".txt"):
            #print(os.path.join(path, file))
            for line in open(os.path.join(path, file),'r'):
                info = line.split()
                #print(info[0]+" - "+info[1]+" - "+info[2]+" - "+info[3]+" - "+info[4])
                #print(info)

                size = float(info[3])*float(info[4])
                ratio = float(info[3])/float(info[4])



                cont[int(info[0])][0] += 1
                cont[int(info[0])][1] += size
                cont[int(info[0])][2] += ratio

                if size < cont[int(info[0])][3]:
                    cont[int(info[0])][3] = size

                if size > cont[int(info[0])][4]:
                    cont[int(info[0])][4] = size

                if ratio < cont[int(info[0])][5]:
                    cont[int(info[0])][5] = ratio

                if ratio > cont[int(info[0])][6]:
                    cont[int(info[0])][6] = ratio


    for i in range(nclases):
        if cont[i][0]>0:
            cont[i][1] = cont[i][1]/cont[i][0]
            cont[i][2] = cont[i][2]/cont[i][0]

        print(cont[i])
        a= " ".join(str(x) for x in cont[i])
        info_file.write(a+ '\n' )

    info_file.close()
    #print cont