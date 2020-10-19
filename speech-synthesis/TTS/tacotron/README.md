# Tacotron

This is a NNabla implementation of the [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135).

All hyper-parameters are defined in [hparams.py](./hparams.py). We use the same values from https://github.com/keithito/tacotron/ as reference.

<img src="./images/o_att.png" width=30% height=30% > <img src="./images/o_mel.png" width=30% height=30% > <img src="./images/o_mag.png" width=30% height=30% >

## Dataset
Run the following commands to prepare the [LJ dataset](https://keithito.com/LJ-Speech-Dataset/),
```bash
bash scripts/prepare_dataset.sh
```
The data will be located into `./data/LJSpeech-1.1/`. There will be three files: `metadata_train.csv`, `metadata_valid.csv`, and `metadata_test.csv`. These files are used for training, validation, and test.

## Requirments
### Python environment
You can set up python dependencies from [requirements.txt](./requirements.txt):

```bash
pip install -r ./requirements.txt
```
Note that this requirements.txt dose not contain `nnabla-ext-cuda`.
If you have CUDA environment, we highly recommend to install `nnabla-ext-cuda` and use GPU devices.
See [NNabla CUDA extension package installation guild](https://nnabla.readthedocs.io/en/latest/python/pip_installation_cuda.html).


## Train
```bash
python train.py --device-id <device id> \
                --context "cudnn"
```
If you have multiple GPUs, then 
```bash
mpirun -n <number of GPUs> python main.py \
    --device-id <list of GPUs>
    --context "cudnn"
```
## Test
```bash
python synthesize.py --device-id <device id> \
    --context "cudnn" \
    --f-model <model file> \
    --f-text <text file> \
    --f-output <output file>
```

The pretrained model can be downloaded from [here]().

Synthesized audio samples can be downloaded from [here]().

# References

1. https://google.github.io/tacotron/
2. https://github.com/keithito/tacotron/
3. [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135)