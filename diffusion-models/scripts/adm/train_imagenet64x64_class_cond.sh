# configs
MODEL="model=adm_64_class_cond"
DATASET="dataset=imagenet"
DIFFUSION="diffusion=cosine"
TRAIN="train=imagenet_64_adm train.output_dir=./logdir/adm_imagent64_class_cond"

# run script
mpirun -N 8 python train.py ${MODEL} ${DATASET} ${DIFFUSION} ${TRAIN}
