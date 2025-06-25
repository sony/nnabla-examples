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


from typing import Any, Union, List
from io import BytesIO
import numpy as np
from PIL import Image
from tqdm import tqdm

import nnabla as nn

from .simple_tokenizer import SimpleTokenizer as _Tokenizer


from .model import encode_text as m_encode_text
from .model import encode_image as m_encode_image
from .model import logits as m_logits

BICUBIC = Image.BICUBIC

__all__ = ["tokenize",
           "load",
           "encode_text",
           "encode_image",
           "preprocess",
           "logits",
           'CLIP',
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

    params = nn.get_parameters()

    vision_patch_size = params["visual/conv1/W"].shape[-1]
    grid_size = round(
        (params["visual/positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size


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
    grid_size = round(
        (params["visual/positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    i_w, i_h = image.size
    if i_w >= i_h:
        res_w = image_resolution * i_w // i_h
        res_h = image_resolution
    else:
        res_w = image_resolution
        res_h = image_resolution * i_h // i_w

    image = image.resize((res_w, res_h), BICUBIC)

    crop_left = int(round((res_w - image_resolution) / 2.))
    crop_top = int(round((res_h - image_resolution) / 2.))
    image = image.crop((crop_left,
                       crop_top,
                       crop_left + image_resolution,
                       crop_top + image_resolution
                        ))

    image = image.convert("RGB")
    image = np.array(image)

    image = _normalize(image, mean, std)
    image = image.transpose((2, 0, 1))

    return image


def encode_text(x):
    return m_encode_text(x)


def encode_image(x):
    return m_encode_image(x)


def logits(image, text):
    return m_logits(image, text)


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
    result = np.zeros((len(all_tokens), context_length), dtype=np.longlong)

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


class CLIP(nn.Module):
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = 'data/ViT-L-14.h5'
        load(model_path)

    def encode_image(self, image: nn.Variable) -> nn.Variable:
        assert image.ndim == 4 and image.shape[
            1] == 3, f'Image shape must be (N, C, H, W). Given {image.shape}'
        return encode_image(image)

    def encode_text(self, text: nn.Variable) -> nn.Variable:
        assert text.ndim == 2, 'Text must be (#texts, #tokens)'
        return encode_text(text)

    def probabilities(self, image_features: nn.Variable, text_features: nn.Variable) -> nn.Variable:
        import nnabla.functions as F
        from .model import cosine_similarity
        logits_per_image = cosine_similarity(image_features, text_features)
        probs = F.softmax(logits_per_image, axis=-1)
        return probs

    def __call__(self, image: nn.Variable, text: nn.Variable) -> nn.Variable:
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        probs = self.probabilities(image_features, text_features)
        return probs

    def preprocess_image(self, image_path: Union[str, BytesIO]) -> nn.Variable:
        from PIL import Image
        img = Image.open(image_path)
        img = preprocess(img)
        img = nn.Variable.from_numpy_array(img[None])
        return img

    def preprocess_texts(self, texts: List[str]) -> nn.Variable:
        tokens = tokenize(texts)
        tokens = nn.Variable.from_numpy_array(tokens)
        return tokens

    def run(self, image_path: Union[str, BytesIO], texts: List[str]) -> np.ndarray:
        img = self.preprocess_image(image_path)
        tokens = self.preprocess_texts(texts)
        with nn.auto_forward(), nn.no_grad():
            probs = self(img, tokens)
        return probs.data.get_data('r')[0]
