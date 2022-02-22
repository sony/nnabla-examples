# FastSpeech 2: Fast and High-Quality End-to-End Text to Speech
This is a NNabla implementation of the [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558). FastSpeech 2 addresses the issues in FastSpeech and solves the one-to-many mapping problem in Text-to-Speech (TTS). It  introduces more variation information of speech (e.g., pitch, energy and more accurate duration) as conditional inputs.


<center>
<img src="./images/fastspeech2.png" width=90% height=90% >
</center>


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
Download the [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) data set and change `corpus_path` to the corresponding directory containing `../LJSpeech-1.1/`.
Adjust `precomputed_path` to the desired directory, which stores all precomputed values.

Alignments for the LJSpeech is provided [here](https://drive.google.com/drive/folders/1DBRkALpPd6FL9gjHMmMEdHODmkgNIIK4?usp=sharing). Download and unzip this file
in `precomputed_path/TextGrid/LJSpeech/`.

Preprocessing step can be done by running the following command
```bash
python preprocess.py
```
It takes around 10 minutes.

## Train
All hyper-parameters used for training are defined at [hparams.py](hparams.py). These parameters can also be changed in the command line.
```bash
mpirun -n <number of GPUs> python main.py -c cudnn -d <list of GPUs e.g., 0,1,2,3> \
       --output_path log/baseline/ \
       --batch_size 8 \
       ...
```

## Inference
To synthesize mel-spectrogram from text, run the following script.
```bash
python synthesize.py -c cudnn -d <list of GPUs e.g., 0,1,2,3> \
       --model <path to pretrained model> \
       --text  <input text> \
       --pitch <pitch control factor> \
       --energy <energy control factor> \
       --duration <duration control factor> \
       --output <output mel-spectrogram file>
```

Note that the outputs are mel spectrograms. Use a vocoder such as [HiFi-GAN](https://github.com/sony/nnabla-examples/tree/master/speech-synthesis/HiFiGAN/) in order to generate audio waveforms from mel spectrograms.


The pre-trained model can be downloaded from [here](https://nnabla.org/pretrained-models/nnabla-examples/speech-synthesis/TTS/FastSpeech2/model.h5).


Synthesized audio samples can be downloaded from [here](https://nnabla.org/pretrained-models/nnabla-examples/speech-synthesis/TTS/FastSpeech2/samples.7z).


# References
1. https://github.com/ming024/FastSpeech2
2. Ren, Y., Hu, C., Tan, X., Qin, T., Zhao, S., Zhao, Z. and Liu, T.Y., 2020. Fastspeech 2: Fast and high-quality end-to-end text to speech. arXiv preprint arXiv:2006.04558.
