#CUDA_VISIBLE_DEVICES=0 python train.py -c config/camvid_segmentation.py -e baseline_camvid_fcn8 -l /home/master/experiments -s /home/master/experiments


CUDA_VISIBLE_DEVICES=0 python train.py -c config/kitti_segmentation.py -e kitti_camvid_fcn8 -l /home/master/experiments -s /home/master/experiments

CUDA_VISIBLE_DEVICES=0 python train.py -c config/cityscapes_segmentation.py -e cityscapes_camvid_fcn8 -l /home/master/experiments -s /home/master/experiments