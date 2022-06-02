# configs
TRAIN_ARGS="--batch-size 4 --accum 8 --type-config float --lr 1e-4 --dataset imagenet"
DIFFUSION_ARGS="--beta-strategy linear --model-var-type learned_range"
CONFIG="--no-progress --show-interval 10 --output-dir ./logdir --gen-interval 20000"
MODEL_CONFIG="--ssn --attention-resolutions 8,16,32 --image-size 256 --num-attention-head-channels 64 --base-channels 256 --resblock-resample"

# run script
mpirun -N 4 python train.py $(echo $TRAIN_ARGS) $(echo $DIFFUSION_ARGS) $(echo $CONFIG) $(echo $MODEL_CONFIG)
