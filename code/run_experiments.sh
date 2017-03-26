CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_ssd.py -e imagenet_ssd -l /home/master/experiments -s /home/master/experiments

CUDA_VISIBLE_DEVICES=0 python train.py -c config/udacity_detection.py -e imagenet_ssd -l /home/master/experiments -s /home/master/experiments

