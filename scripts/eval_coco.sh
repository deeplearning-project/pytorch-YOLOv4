#python ./utils/evaluate_on_coco.py  -g 5 -dir /dataset/yangfan/coco/images/val2017 -gta /dataset/yangfan/coco/annotations/instances_val2017.json --load ./models/Yolov4_epoch30.pth -ld './logs/eval' -c ./cfg/yolov4.cfg 
python ./utils/evaluate_on_coco.py --is-SE -g 5 -dir /dataset/yangfan/coco/images/val2017 -gta /dataset/yangfan/coco/annotations/instances_val2017.json --load ./models/yolo_resnet50_SE/Yolov4_epoch31.pth -ld './logs/eval' --resnet-name 50 
