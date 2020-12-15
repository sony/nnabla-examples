# StyleGAN2 Distillation model

## To generate the dataset for training 

Download precomputed stylegan2 directions zip file from [https://hostb.org/NCM](here) and unzip the downloaded file. After downloading, dataset can be created by running: 

`$ python data_generation.py --save-image-path [path to save the images] --attr-delta-path [path to the attribute npy file in the downloaded folder] --weights-path [path to store the stylegan2 pretrained weights] --num-image [number of images to generate] --batch-size [batch-size for single forward pass of the stylegan2 generator] --coeff [for controlling the amount of modification of the attribute] ` 

## To train Pix2PixHD model for the image translation task on the dataset

For training the global generator (trains on 512x512 images):

For single gpu: 

`python main.py --data-root [path to the dataset]`

For multiple gpus (say 4):

`mpirun -N 4 python main.py --data-root`

After training the global generator, local generator can be trained as: 

For single gpu: 

`python main.py --fix-global-epoch 20 --d-n-scales 3 --g-n-scales 2 --save-path --load-path [path to trained global generator parameters (h5 file)]`

For multiple gpus (say 4):

`mpirun -N 4 python main.py --fix-global-epoch 20 --d-n-scales 3 --g-n-scales 2 --save-path --load-path [path to trained global generator parameters (h5 file)]`

## References
[https://github.com/EvgenyKashin/stylegan2-distillation](https://github.com/EvgenyKashin/stylegan2-distillation)
