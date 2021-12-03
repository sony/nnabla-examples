# configs
TRAIN_ARGS="--batch-size 4 --beta-strategy linear --dataset celebahq --dataset-on-memory"
CONFIG="--no-progress --show-interval 10 --output-dir ./logdir --gen-interval 20000"
MODEL_CONFIG="--ssn --num-attention-heads 4 --attention-resolutions 8,16 --image-size 256"

# run script
mpirun -N 4 python train.py $(echo $TRAIN_ARGS) $(echo $CONFIG) $(echo $MODEL_CONFIG)
