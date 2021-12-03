# configs
TRAIN_ARGS="--batch-size 32 --beta-strategy cosine --dataset imagenet --dataset-on-memory"
CONFIG="--no-progress --show-interval 10 --output-dir ./logdir --gen-interval 20000"
MODEL_CONFIG="--ssn --num-attention-heads 4 --attention-resolutions 8,16 --image-size 64"

# run script
mpirun -N 4 python train.py $(echo $TRAIN_ARGS) $(echo $CONFIG) $(echo $MODEL_CONFIG)
