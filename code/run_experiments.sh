#CUDA_VISIBLE_DEVICES=0 python train.py -c config/test1.py -e resize_256_crop_224 -l /home/master/experiments -s /home/master/experiments

#CUDA_VISIBLE_DEVICES=0 python train.py -c config/test2.py -e norm_featurewise_centerDEBUG -l /home/master/experiments -s /home/master/experiments

#CUDA_VISIBLE_DEVICES=0 python train.py -c config/wideresnet.py -e wide_resnet_scratch -l /home/master/experiments -s /home/master/experiments

#sleep 5

#CUDA_VISIBLE_DEVICES=0 python train.py -c config/wideresnet_imagenet.py -e wide_resnet_imagenet -l /home/master/experiments -s /home/master/experiments

#CUDA_VISIBLE_DEVICES=0 python train.py -c config/test3.py -e norm_imagenet -l /home/master/experiments -s /home/master/experiments

CUDA_VISIBLE_DEVICES=0 python train.py -c config/kitti_scratch.py -e kitti_scratch -l /home/master/experiments -s /home/master/experiments

sleep 5

CUDA_VISIBLE_DEVICES=0 python train.py -c config/kitti_imagenet.py -e kitti_imagenet -l /home/master/experiments -s /home/master/experiments
