# configs
MODEL="model=iddpm_32"
DATASET="dataset=cifar10"
DIFFUSION="diffusion=cosine"
TRAIN="train=cifar10_iddpm"

# run script
mpirun -N 8 python train.py ${MODEL} ${DATASET} ${DIFFUSION} ${TRAIN}