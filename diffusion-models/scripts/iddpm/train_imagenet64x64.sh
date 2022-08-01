# configs
MODEL="model=iddpm_64"
DATASET="dataset=imagenet"
DIFFUSION="diffusion=cosine"
TRAIN="train=imagenet_64_iddpm"

# run script
mpirun -N 8 python train.py ${MODEL} ${DATASET} ${DIFFUSION} ${TRAIN}
