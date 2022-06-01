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


import nnabla as nn
import nnabla.functions as F

import numpy as np


def get_logits(image_features, text_features, aggregate=False, checking=nn.Variable()):

    # normalized features
    image_features = image_features / \
        F.norm(image_features, axis=1, keepdims=True)
    text_features = text_features / \
        F.norm(text_features, axis=1, keepdims=True)

    # cosine similarity as logits
    logit_scale = nn.parameter.get_parameter_or_create(
        name='logit_scale', shape=())
    checking = logit_scale
    logit_scale = F.mean(F.exp(logit_scale))

    image_features = image_features.reshape(
        (1, image_features.shape[0], image_features.shape[1]))
    text_features = F.transpose(text_features, (1, 0))
    text_features = text_features.reshape(
        (1, text_features.shape[0], text_features.shape[1]))

    per_image = F.batch_matmul(image_features, text_features).reshape(
            (image_features.shape[1], image_features.shape[1])
        )  # shape = [global_batch_size, global_batch_size]
    logits_per_image = logit_scale.reshape((1, 1)) * per_image

    logits_per_text = F.transpose(logits_per_image, (1, 0))

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text, checking
