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


import numpy as np
import pandas as pd
import nnabla as nn
import nnabla.functions as F
from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator, data_iterator_simple
from nnabla.utils.data_source import DataSource

import glob
from PIL import Image


# Data interator for CLIP
# preprocess for image and text
# return [image, text]
def clip_data_iterator(clip, txt_path, batch_size=1, image_size=224, num_samples=-1,
                       normalize_method=lambda x: (x - 127.5) / 127.5,
                       shuffle=True,
                       image_key="filepath",
                       caption_key="title",
                       rng=None):
    # Load txt file (image_path and ception)
    df = pd.read_csv(txt_path, sep="\t")
    images = df[image_key].tolist()
    captions = df[caption_key].tolist()

    if num_samples == -1:
        assert len(images) == len(
            captions), "The number of images and captions must be same"
        num_samples = len(images)
    else:
        logger.info(
            "Num. of data ({}) is used for debugging".format(num_samples))

    def load_func(i):
        text = captions[i]
        # Note: change prefix as per your environment
        img_path = "/conceptual_captions/cc_3m/" + images[i]
        image = clip.preprocess(Image.open(img_path))
        # shape [num_text_per_image, 77] but in training phase, num_text_per_image=1
        text = clip.tokenize(text, truncate=True)
        text = np.reshape(text, (77,))

        # Return numpy.ndarray, not Variable
        return image, text

    return data_iterator_simple(load_func, num_samples, batch_size,
                                shuffle=shuffle, rng=rng, with_file_cache=False)


if __name__ == '__main__':
    # Hand-made test
    import clip
    clip.load("asset/ViT-B-32.h5")
    txt_path = "asset/minisample.txt"
    loader = clip_data_iterator(clip, txt_path, batch_size=1, image_size=256)

    for _ in range(4):
        imgs, texts = loader.next()
        print(imgs[0].shape, texts[0])
