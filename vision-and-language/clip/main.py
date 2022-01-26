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
from PIL import Image

import clip

def main():
    with nn.auto_forward():
        clip.load('data/ViT-B-32.h5')

        image = clip.preprocess(Image.open("CLIP.png"))
        text = clip.tokenize(["a diagram", "a dog", "a cat"])

        image_features = clip.encode_image(image)
        text_features = clip.encode_text(text)
            
        logits_per_image, logits_per_text = clip.logits(image, text)
        probs = F.softmax(logits_per_image, axis=-1)

        print("Label probs:", probs.d)  # prints: [[0.9927937  0.00421068 0.00299572]]


if __name__ == "__main__":
    main()