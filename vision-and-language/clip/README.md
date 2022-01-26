# NNabla-CLIP

This repository is a nnabla implementation of CLIP released by openAI.
Click [here](https://arxiv.org/abs/2103.00020) for the original paper.

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
    text = clip.tokenize(["a diagram", "a dog", "a cat"])

    image_features = clip.encode_image(image)
    text_features = clip.encode_text(text)
            
    logits_per_image, logits_per_text = clip.logits(image, text)
    probs = F.softmax(logits_per_image, axis=-1)

    print("Label probs:", probs.d)  # prints: [[0.9927937  0.00421068 0.00299572]]
```

## API

The CLIP module `clip` provides the following methods:

#### `clip.available_models()`

Returns the names of the available CLIP models.

#### `clip.load(path)`

 Loads the model specified by the path to the trained model file.

#### `clip.tokenize(text: Union[str, List[str]], context_length=77)`

Returns a NdArray containing tokenized sequences of given text input(s). This can be used as the input to the model

#### `clip.encode_image(image: numpy.ndarray)`

Given a batch of images, returns the image features encoded by the vision portion of the CLIP model.

#### `clip.encode_text(text: numpy.ndarray)`

Given a batch of text tokens, returns the text features encoded by the language portion of the CLIP model.

#### `clip.logits(image: numpy.ndarray, text: numpy.ndarray)`

Given a batch of images and a batch of text tokens, returns two ndarrays, containing the logit scores corresponding to each image and text input. The values are cosine similarities between the corresponding image and text features, times 100.