# python utils/train_resnet.py --net-name ResNet101 --gpu-ids 6,7 --batch-size 256 
OMP_NUM_THREADS=2 python utils/train_resnet.py --net-name ResNet50_SE --gpu-ids 0 --batch-size 64 --nThreads 2 --log-dir ./logs/backbone/ResNet50_SE --tensor-dir ./tensor_logs/backbone/ResNet50_SE --save-dir ./models/backbone --start 47 
#OMP_NUM_THREADS=2 python utils/train_resnet.py --net-name ResNet50 --gpu-ids 1 --batch-size 64 --nThreads 2 --log-dir ./logs/backbone/ResNet50 --tensor-dir ./tensor_logs/backbone/ResNet50 --save-dir ./models/backbone --start 4
