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


import nnabla as nn
import nnabla.functions as F
from nnabla.ext_utils import get_extension_context
from PIL import Image

import clip

ctx = get_extension_context('cudnn')
nn.set_default_context(ctx)


def main():
    with nn.auto_forward():
        clip.load('data/ViT-B-32.h5')

        image = nn.Variable.from_numpy_array(
            clip.preprocess(Image.open("CLIP.png")))
        image = F.reshape(
            image, (1, image.shape[0], image.shape[1], image.shape[2]))
        text = nn.Variable.from_numpy_array(
            clip.tokenize(["a diagram", "a dog", "a cat"]))

        image_features = clip.encode_image(image)
        text_features = clip.encode_text(text)

        logits_per_image, logits_per_text = clip.logits(image, text)
        probs = F.softmax(logits_per_image, axis=-1)

        # prints: [[0.9927937  0.00421068 0.00299572]]
        print("Label probs:", probs.d)


if __name__ == "__main__":
    main()
