# Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis

---

## Overview

This is an NNabla implementation of ["Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis."](https://openreview.net/forum?id=1Fqg133qRaI)

These are result examples:

![result examples](./example.png)

---

## Start training

Prepare a training dataset. Several datasets for few-shot learning of GANs are available [here](https://hanlab.mit.edu/projects/data-efficient-gans/datasets/).

Once the dataset is ready, you can start training by

```
python train.py --img-path *path-to-dataset*
```

### Pre-trained Weights
|Dataset | Weights |
|---|---|
|[Bridge of Sighs](https://hanlab.mit.edu/projects/data-efficient-gans/datasets/100-shot-bridge_of_sighs.zip)|[Dis](https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/bridge/Dis_iter100000.h5) [Gen](https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/bridge/Gen_iter100000.h5) [Gen(EMA)](https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/bridge/GenEMA_iter100000.h5)|
|[Grumpy Cat](https://hanlab.mit.edu/projects/data-efficient-gans/datasets/100-shot-grumpy_cat.zip)|[Dis](https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/cat/Dis_iter100000.h5) [Gen](https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/cat/Gen_iter100000.h5) [Gen(EMA)](https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/cat/GenEMA_iter100000.h5)|
|[Medici Fountain](https://hanlab.mit.edu/projects/data-efficient-gans/datasets/100-shot-medici_fountain.zip)|[Dis](https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/fountain/Dis_iter100000.h5) [Gen](https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/fountain/Gen_iter100000.h5) [Gen(EMA)](https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/fountain/GenEMA_iter100000.h5)|
|[Obama](https://hanlab.mit.edu/projects/data-efficient-gans/datasets/100-shot-obama.zip)|[Dis](https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/obama/Dis_iter100000.h5) [Gen](https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/obama/Gen_iter100000.h5) [Gen(EMA)](https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/obama/GenEMA_iter100000.h5)|
|[Panda](https://hanlab.mit.edu/projects/data-efficient-gans/datasets/100-shot-panda.zip)|[Dis](https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/panda/Dis_iter100000.h5) [Gen](https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/panda/Gen_iter100000.h5) [Gen(EMA)](https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/panda/GenEMA_iter100000.h5)|
|[Temple of Heaven](https://hanlab.mit.edu/projects/data-efficient-gans/datasets/100-shot-temple_of_heaven.zip)|[Dis](https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/temple/Dis_iter100000.h5) [Gen](https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/temple/Gen_iter100000.h5) [Gen(EMA)](https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/temple/GenEMA_iter100000.h5)|
|[Wuzhen](https://hanlab.mit.edu/projects/data-efficient-gans/datasets/100-shot-wuzhen.zip)|[Dis](https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/wuzhen/Dis_iter100000.h5) [Gen](https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/wuzhen/Gen_iter100000.h5) [Gen(EMA)](https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/wuzhen/GenEMA_iter100000.h5)|
|[Dog](https://hanlab.mit.edu/projects/data-efficient-gans/datasets/AnimalFace-dog.zip)|[Dis](https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/dog/Dis_iter100000.h5) [Gen](https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/dog/Gen_iter100000.h5) [Gen(EMA)](https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/dog/GenEMA_iter100000.h5)|


## Inference

You can generate images using pre-trained weights by

```
python generate.py --model-load-path *directory that contains the parameters*
```

Note that it will generate two images, one in a regular way, and another with estimated moving average (EMA).

## Interpolation and style mixing

You can also try interpolation and style mixing of generated images.

```
python interpolate.py --model *path to model parameters* --out *file name to save*

python stylemix.py --model *path to model parameters* --out *file name to save*
```


---

## License

See `LICENSE`.
