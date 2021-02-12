# WaveGlow

This is a NNabla implementation of the [WaveGlow: A Flow-based Generative Network for Speech Synthesis](https://arxiv.org/pdf/1811.00002).

WaveGlow is a flow-based network capable of generating high quality speech from mel spectrograms. It is trained by maximizing the likelihood of the training data.

All hyper-parameters are defined in [hparams.py](./hparams.py). We use the values from https://github.com/NVIDIA/waveglow as reference. Note that number of residual channels in the coupling layer networks is 256 instead of 512.



## Requirements
### Python environment
Install `python >= 3.6`, then set up python dependencies from [requirements.txt](./requirements.txt):

```bash
pip install -r ./requirements.txt
```
Note that this requirements.txt dose not contain `nnabla-ext-cuda`.
If you have CUDA environment, we highly recommend to install `nnabla-ext-cuda` and use GPU devices.
See [NNabla CUDA extension package installation guide](https://nnabla.readthedocs.io/en/latest/python/pip_installation_cuda.html).

## Dataset
Run the following commands to prepare the [LJ dataset](https://keithito.com/LJ-Speech-Dataset/),
```bash
bash scripts/prepare_dataset.sh
```
This will take approximately 30 minutes. The data will be located into `./data/LJSpeech-1.1/`. There will be two files: `metadata_train.csv` and `metadata_valid.csv`. These files are used for training and validation, respectively.

## Train
```bash
python main.py --device-id <device id> \
                --context "cudnn"
```
If you have multiple GPUs, then 
```bash
mpirun -n <number of GPUs> python main.py \
    --device-id <list of GPUs>
    --context "cudnn"
```
Expected training time on 4 GeForce RTX 2080 Ti is 8.5 days.

## Inference
Download mel-spectrogram samples from [here](https://nnabla.org/pretrained-models/nnabla-examples/speech-synthesis/WaveGlow/mel_samples.7z).

Run the following command to synthesize audios from mel spectrograms.

```bash
python synthesize.py --device-id <device id> \
    --context "cudnn" \
    --f-model <model file> \
    --f-mel <mel file> \
    --f-output <output file>
```

The pre-trained model can be downloaded from [here](https://nnabla.org/pretrained-models/nnabla-examples/speech-synthesis/WaveGlow/model.h5).



Synthesized audio samples can be downloaded from [here](https://nnabla.org/pretrained-models/nnabla-examples/speech-synthesis/WaveGlow/samples.7z).

# References
1. https://github.com/NVIDIA/waveglow
2. Prenger, R., Valle, R. and Catanzaro, B., 2019, May. [Waveglow: A flow-based generative network for speech synthesis](https://arxiv.org/abs/1811.00002). In ICASSP, pp. 3617-3621.