# configs
TRAIN_ARGS="--batch-size 32 --beta-strategy cosine --dataset cifar10 --dropout 0.1"
CONFIG="--no-progress --show-interval 10 --output-dir ./logdir --gen-interval 20000"

# run script
mpirun -N 4 python train.py $(echo $TRAIN_ARGS) $(echo $CONFIG)
