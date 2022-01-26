# Copyright 2021 Sony Group Corporation
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


import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List

import numpy as np
from PIL import Image
from tqdm import tqdm

import nnabla as nn

import albumentations as A

from .model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer


from .model import encode_text as m_encode_text
from .model import encode_image as m_encode_image
from .model import logits as m_logits

BICUBIC = Image.BICUBIC

__all__ = ["available_models",
            "tokenize", 
            "load", 
            "encode_text",
            "encode_image", 
            "preprocess",
            "logits",
            ]
_tokenizer = _Tokenizer()

_MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
}

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

    params = nn.get_parameters()

    # vision_width = state_dict["visual.conv1.weight"].shape[0]
    # vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = params["visual/conv1/W"].shape[-1]
    grid_size = round((params["visual/positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    return build_model()

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
    
    params = nn.get_parameters()
    vision_patch_size = params["visual/conv1/W"].shape[-1]
    grid_size = round((params["visual/positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    
    i_w, i_h = image.size
    res_w = image_resolution * i_w // i_h

    image = image.resize((res_w, image_resolution), BICUBIC)

    crop_left = int(round((res_w - image_resolution) / 2.))
    crop_top = int(round((image_resolution - image_resolution) / 2.)) # const 0 in this case
    image = image.crop((crop_left,
                       crop_top,
                       crop_left + image_resolution,
                       crop_top + image_resolution
                       ))
    
    image = image.convert("RGB")
    image = np.array(image)
    
    image = _normalize(image, mean, std)
    
    return image.transpose((2, 0, 1))

def encode_text(x):
    return m_encode_text(x)

def encode_image(x):
    return m_encode_image(x)

def logits(image, text):
    return m_logits(image, text)

def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


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
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = np.zeros((len(all_tokens), context_length), dtype=np.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = np.array(tokens)

    return result