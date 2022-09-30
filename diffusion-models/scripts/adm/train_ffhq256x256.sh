# configs
MODEL="model=adm_256"
DATASET="dataset=ffhq256 dataset.on_memory=True dataset.data_dir=./data/FFHQ_256/"
DIFFUSION="diffusion=linear"
TRAIN="train=ffhq_256_adm train.output_dir=./logdir/ffhq_256_adm"

# run script
mpirun -N 8 python train.py ${MODEL} ${DATASET} ${DIFFUSION} ${TRAIN}