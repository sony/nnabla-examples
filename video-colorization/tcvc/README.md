# TCVC Example by NNabla
This is a NNabla implementation of the "[Automatic Temporally Coherent Video Colorization (TCVC)](https://arxiv.org/abs/1904.09527)".

This work is done by Mingcheng Yuan during his internship at Sony R&D in 2021

<br/>

## Requirements
### Python Environments
You can install all dependencies from `requirements.txt`.
```bash
pip install -r ./requirements.txt
``` 

Note that `nnabla-ext-cuda110`(requiring CUDA 11.0.3 and CUDNN 8.0.5) will be installed in default settings.  
If you've installed other version of CUDA, please modify `requirements.txt` to meet your environment.  
For more information, see [NNabla CUDA extension package installation guild](https://nnabla.readthedocs.io/en/latest/python/pip_installation_cuda.html).

<br/>

## Dataset
Please prepare the dataset by extracting frames from anime episodes (users should prepare their own arbitrary anime clips)  
Extract with the following command (requires `ffmpeg`, currently won't work when installed with pip but works with conda)
```
python -m external.extract_video_frames --input-path path-to-video 
```
After extracting the frame, extract the line-art with "[sketchKeras](https://github.com/lllyasviel/sketchKeras)"

<br/>

## Running Code
### Train with the original tcvc ( _feat for reference-based version)
```
# by a single GPU
./scripts/train_tcvc.sh
or
./scripts/train_tcvc_ref.sh

# by multi GPUs
./scripts/train_tcvc_mgpu.sh
or
./scripts/train_tcvc_ref_mgpu.sh
```
Trained model will be saved under the log directory (default is `./result/tcvc_01/`).
Tensorboard logs will be saved under `./result/tcvc_01/tb_logs`.

### Generate Image 
```
# original tcvc 
./scripts/generate_tcvc.sh

# reference-based version
./scripts/generate_tcvc_ref.sh
```

In the default setting, model parameter is loaded from `./results/tcvc_01/015_param.h5` which is created at the end of 16th epoch during training script.  
Generated images will be saved under the log directory (default is `./results/tcvc_01/gen_results`).

<br/>

## Note 
* All configurations are listed in config.yaml. You can change model settings by editing this.

<br/>

## Reference
- Thasarathan, Harrish, Kamyar Nazeri, and Mehran Ebrahimi,  "[Automatic Temporally Coherent Video Colorization](https://github.com/iver56/automatic-video-colorization)"

