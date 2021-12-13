# StyleGAN2 
This is the implementation of StyleGAN-2 in nnabla. The implementation has been tested for [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset. 

## Instructions to run:

### Training

After setting the path to the FFHQ dataset and various other hyperparameters in `configs/ffhq.yaml`, the training script can be executed as:

```
python main.py --img_size [256/512/1024] --batch_size [8/4/1] --train --results_dir [path to save intermediate generator results] --weights_path [path to save trained model weights]
```

Some arguments can also be passed from the command line. These are as follows:
- `--device_id [id]` to train on a specific GPU. Same effect as the execution instruction prefix `CUDA_VISIBLE_DEVICES=[id]`. (`default=0`)
- `--auto_forward` to run the execution in auto-forward or dynamic computation mode (`default=False`)
- `--extension_module [cuda/cudnn/cpu]` to set the extension module for nnabla (`default='cudnn'`)
- `--dali`: To use [DALI](https://github.com/NVIDIA/DALI) based data iterator for fetching the data (`default=False`)

Please note that StyleGAN2 training is very time-consuming. Training on 50000 FFHQ images (Image resoultion 512x512) on 4 Nvidia TITAN RTX gpus for 70 epochs takes over 12 days. 

### Inference

Inference operations of image generation, latent space interpolation in W space, style mixing, latent space projection and perceptual path length calculation is supported. 
For inferece, run as follows:

```
python main.py --img_size [256/512/1024] --test [generate/latent_space_interpolation/style_mixing/latent_space_projection/ppl] --weights_path [PATH to pretrained weights] --results_dir [path to store generated results]
```

To infer with the weights from the original StyleGAN-2 repository, run as follows:

```
python main.py --img_size [256/512/1024] --test [generate/latent_space_interpolation/style_mixing/latent_space_projection/ppl] --use_tf_weights --results_dir [path to store generated results] 
```

Following set of additional arguments from command line are supported:

- For `generate`: 
  - `--batch_size [BATCH-SIZE]`: in this case, number of images to generate

- For `latent_space_interpolation`: 
  - `--batch_size [BATCH-SIZE]`: in this case, number of images to generate
  - `--seed_1`: Seed for random sampling of first latent vector in Z space
  - `--seed_2`: Seed for random sampling of second latent vector in Z space

- For `style_mixing`: 
  - `--batch_size_A [BATCH-SIZE]`: Batch-size for style A: coarse style
  - `--batch_size_B [BATCH-SIZE]`: Batch-size for style B: fine style

- For `latent_space_projection`: 
  - `--img_path [PATH]`: Path to the image to be projected into the latent space

- For `ppl`: 
  - `--batch_size [BATCH-SIZE]`: in this case, number of images to generate in one forward pass of the evaluation (Note that FID and IS are not supported for now)

## References
- [StyleGAN2 paper](http://arxiv.org/abs/1912.04958)
- [StyleGAN2 official (tensorflow) implementation](https://github.com/NVlabs/stylegan2)
- [StyleGAN2 pytorch implementation](https://github.com/rosinality/stylegan2-pytorch) 
