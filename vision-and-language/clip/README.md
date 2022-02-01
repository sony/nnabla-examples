# NNabla-CLIP

This repository contains the nnabla implementation of CLIP(Contrastive Language-Image Pre-Training) released by openAI.  
Click [here](https://arxiv.org/abs/2103.00020) for the original paper "Learning Transferable Visual Models From Natural Language Supervision".

## Approach

![CLIP](CLIP.png)

## Usage

you can setup the environment via following command:
```bash
$ pip install -r requirements.txt
```

```python
import nnabla as nn
import nnabla.functions as F
from PIL import Image

import clip

with nn.auto_forward():
    clip.load('data/ViT-B-32.h5')

    image = clip.preprocess(Image.open("CLIP.png"))
    image = F.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
    text = clip.tokenize(["a diagram", "a dog", "a cat"])

    image_features = clip.encode_image(image)
    text_features = clip.encode_text(text)
            
    logits_per_image, logits_per_text = clip.logits(image, text)
    probs = F.softmax(logits_per_image, axis=-1)

    print("Label probs:", probs.d)  # prints: [[0.9927937  0.00421068 0.00299572]]
```

## Download trained models

Trained models are available as:
- [ViT-B/32](https://drive.google.com/file/d/1I_A4esqGGDSuSu1-VrjTvPjxV52WB62A/view?usp=sharing)(default) to `data/`
- [ViT-B/16](https://drive.google.com/file/d/1M_9wXEXjuRwSe3Zcdn9gFrMmtkRyg3Qm/view?usp=sharing) to `data/`
- [ViT-L/14](https://drive.google.com/file/d/1n9R0uXvS9fLVMUjEtkMYwdd4PmcZ7gFq/view?usp=sharing) to `data/`  

You can also refer to the [conversion code](notebook-examples/convert_weights.ipynb) in case you'd like to know how PyTorch weights are converted to nnabla.

## API

The CLIP module `clip` provides the following methods:

#### `clip.load(path)`

 Loads the model specified by the path to the trained model file.

 #### `clip.preprocess(image: PIL Image)`
 Returns a Variable pre-processed for input to the image encoder.

#### `clip.tokenize(text: Union[str, List[str]], context_length=77)`

Returns a Variable containing tokenized sequences of given text input(s). This can be used as the input to the model.

#### `clip.encode_image(image: Variable)`

Given a batch of images, returns the image features encoded by the vision portion of the CLIP model.  
The input shape must be ***(batch_size, c, h, w)***

#### `clip.encode_text(text: Variable)`

Given a batch of text tokens, returns the text features encoded by the language portion of the CLIP model.

#### `clip.logits(image: numpy.ndarray, text: numpy.ndarray)`

Given a batch of images and a batch of text tokens, returns two ndarrays, containing the logit scores corresponding to each image and text input. The values are cosine similarities between the corresponding image and text features, times 100.

#### Contributor
This work was done during [Soma Kanazawa's](https://github.com/soma-knzw) internship at Sony.