# configs
MODEL="model=adm_256"
DATASET="dataset=celebahq dataset.on_memory=True dataset.data_dir=./data/celeba-hq-512"
DIFFUSION="diffusion=linear"
TRAIN="train=imagenet_256_adm train.output_dir=./logdir/celebahq_256_adm"

# run script
mpirun -N 8 python train.py ${MODEL} ${DATASET} ${DIFFUSION} ${TRAIN}
