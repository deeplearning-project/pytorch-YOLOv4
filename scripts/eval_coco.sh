python ./utils/evaluate_on_coco.py  -g 1,2,3 -dir /dataset/yangfan/coco/images/val2017 -gta /dataset/yangfan/coco/annotations/instances_val2017.json --load ./models/Yolov4_epoch30.pth -ld './logs' -c ./cfg/yolov4.cfg 
