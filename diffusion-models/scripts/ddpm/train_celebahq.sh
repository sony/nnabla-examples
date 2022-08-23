# configs
MODEL="model=ddpm_256"
DATASET="dataset=celebahq dataset.on_memory=True dataset.data_dir=./data/celeba-hq-512"
DIFFUSION="diffusion=linear"
TRAIN="train=celebahq_256_ddpm"

# run script
mpirun -N 8 python train.py ${MODEL} ${DATASET} ${DIFFUSION} ${TRAIN}
