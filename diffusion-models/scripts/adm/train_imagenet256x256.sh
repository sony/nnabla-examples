# configs
MODEL="model=adm_256"
DATASET="dataset=imagenet"
DIFFUSION="diffusion=linear"
TRAIN="train=imagenet_256_adm"

# run script
mpirun -N 8 python train.py ${MODEL} ${DATASET} ${DIFFUSION} ${TRAIN}
