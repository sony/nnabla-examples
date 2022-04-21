# Copyright 2022 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Union, List

import numpy as np
from PIL import Image

import nnabla as nn

from nnabla_cliport.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from nnabla_cliport.clip.model import encode_text as m_encode_text
from nnabla_cliport.clip.model import encode_image as m_encode_image
from nnabla_cliport.clip.model import logits as m_logits

BICUBIC = Image.BICUBIC

__all__ = ["tokenize",
           "load",
           "encode_text",
           "encode_image",
           "preprocess",
           "logits",
           ]
_tokenizer = _Tokenizer()


def _normalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


def load(name, jit=False, download_root=None):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name or the path to the checkopoint 
    jit : bool, optional
        [description], by default False
    download_root : [type], optional
        path to download the model files, by default None
    """
    nn.clear_parameters()
    nn.load_parameters(name)


def preprocess(image):
    """Image Preprocessor

    Parameters
    ----------
    image : PIL PngImageFile
        single image

    Returns
    -------
    image : numpy.ndarray
    """
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)

    image_resolution = 224

    i_w, i_h = image.size
    res_w = image_resolution * i_w // i_h

    image = image.resize((res_w, image_resolution), BICUBIC)

    crop_left = int(round((res_w - image_resolution) / 2.))
    # const 0 in this case
    crop_top = int(round((image_resolution - image_resolution) / 2.))
    image = image.crop((crop_left,
                       crop_top,
                       crop_left + image_resolution,
                       crop_top + image_resolution))

    image = image.convert("RGB")
    image = np.array(image)

    image = _normalize(image, mean, std)
    image = image.transpose((2, 0, 1))

    return image


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False):
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] +
                  _tokenizer.encode(text) + [eot_token] for text in texts]
    result = np.zeros((len(all_tokens), context_length), dtype=np.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = np.array(tokens)

    return result


def encode_text(x):
    return m_encode_text(x)


def encode_image(x):
    return m_encode_image(x)


def logits(image, text):
    return m_logits(image, text)
