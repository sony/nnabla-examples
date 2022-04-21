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

import pathlib

from PIL import Image

import numpy as np

import nnabla as nn
import nnabla_cliport.clip.clip as nnabla_clip
from nnabla_cliport.models.clip_text_encoder import CLIPTextEncoder, CLIPTextEncoderOld
from nnabla_cliport.models.clip_image_encoder import CLIPImageEncoder, CLIPImageEncoderOld


def load_model(filepath):
    nn.clear_parameters()
    nn.load_parameters(filepath)


def convert_image_encoder_model(image, original_params):
    image_encoder = CLIPImageEncoderOld()
    visual_params = {}
    beta = None
    gamma_key = None
    gamma = None
    for key, param in original_params.items():
        if 'visual' in key and 'num_batches_tracked' not in key:
            if ('bn' in key or 'downsample/1' in key) and 'W' in key:
                gamma = param
                gamma_key = key
            elif ('bn' in key or 'downsample/1' in key) and 'b' in key and 'var' not in key and 'mean' not in key:
                beta = param
                visual_params[key] = beta
                visual_params[gamma_key] = gamma
                print(f'old name: {key}, param shape: {param.shape}')
                print(f'old name: {gamma_key}, param shape: {gamma.shape}')
            else:
                print(f'old name: {key}, param shape: {param.shape}')
                visual_params[key] = param
    image_encoder2 = CLIPImageEncoder('image_encoder')
    old_feature = image_encoder.encode_image(image)
    new_feature = image_encoder2.encode_image(image)
    new_params = image_encoder2.get_parameters(grad_only=False)
    for key, param in new_params.items():
        print(f'new name: {key}, param shape: {param.shape}')
    for old_param, (new_key, new_param) in zip(visual_params.values(), new_params.items()):
        print(f'old param shape: {old_param.shape}')
        print(f'new param shape: {new_param.shape}')
        new_param.d = np.reshape(old_param.d, newshape=new_param.shape)
    nn.forward_all([old_feature[0], new_feature[0]])
    print(f'old_feature shape: {old_feature[0].d.shape}')
    print(f'new_feature shape: {new_feature[0].d.shape}')
    print(f'old_feature: {old_feature[0].d}')
    print(f'new_feature: {new_feature[0].d}')
    scriptdir = pathlib.Path(__file__).parent
    image_encoder2.save_parameters(f'{scriptdir}/data/image_encoder.h5')


def split_resblock_keys_to_each_layer(resblock_keys):
    layer_keys = {}
    for resblock_key in resblock_keys:
        layer_number = int(resblock_key.split('/')[2])
        layer_key = layer_keys.get(layer_number, [])
        layer_key.append(resblock_key)
        layer_keys[layer_number] = layer_key
    return layer_keys


def reorder_resblock_keys(resblock_keys):
    layer_num = len(set(k.split('/')[2] for k in resblock_keys))

    reordered_keys = []
    for layer in range(layer_num):
        reordered_keys.append(f'transformer/resblocks/{layer}/ln_1/b')
        reordered_keys.append(f'transformer/resblocks/{layer}/ln_1/W')
        reordered_keys.append(f'transformer/resblocks/{layer}/attn/in_proj_W')
        reordered_keys.append(f'transformer/resblocks/{layer}/attn/in_proj_b')
        reordered_keys.append(f'transformer/resblocks/{layer}/attn/out_proj/W')
        reordered_keys.append(f'transformer/resblocks/{layer}/attn/out_proj/b')
        reordered_keys.append(f'transformer/resblocks/{layer}/ln_2/b')
        reordered_keys.append(f'transformer/resblocks/{layer}/ln_2/W')
        reordered_keys.append(f'transformer/resblocks/{layer}/mlp/c_fc/W')
        reordered_keys.append(f'transformer/resblocks/{layer}/mlp/c_fc/b')
        reordered_keys.append(f'transformer/resblocks/{layer}/mlp/c_proj/W')
        reordered_keys.append(f'transformer/resblocks/{layer}/mlp/c_proj/b')
    return reordered_keys


def convert_text_encoder_model(text, original_params):
    token = nn.Variable.from_numpy_array(nnabla_clip.tokenize(text))
    text_encoder = CLIPTextEncoderOld()
    text_params = {}

    resblock_keys = [key for key in original_params.keys(
    ) if 'resblocks' in key and 'visual' not in key]
    reordered_keys = reorder_resblock_keys(resblock_keys)
    reordered_keys = ['token_embedding/W', 'positional_embedding'] + \
        reordered_keys + \
        ['ln_final/b', 'ln_final/W', 'text_projection']
    text_params = {}
    for key in reordered_keys:
        text_params[key] = original_params[key]

    old_feature = text_encoder.encode_text(token)

    text_encoder2 = CLIPTextEncoder('text_encoder',
                                    embed_dim=1024,
                                    context_length=77,
                                    transformer_layers=12,
                                    transformer_heads=8,
                                    transformer_width=512,
                                    vocab_size=49408)
    new_feature = text_encoder2.encode_text(token)
    new_params = text_encoder2.get_parameters(grad_only=False)
    assert len(text_params) == len(new_params)
    for key, param in new_params.items():
        print(f'new name: {key}, param shape: {param.shape}')
    for old_param, (new_key, new_param) in zip(text_params.values(), new_params.items()):
        print(f'old param shape: {old_param.shape}')
        print(f'new param shape: {new_param.shape}')
        if 'affine/W' in new_key:
            new_param.d = np.transpose(old_param.d, axes=(1, 0))
        else:
            new_param.d = old_param.d
    nn.forward_all([old_feature[0], new_feature[0]])
    print(f'old_feature shape: {old_feature[0].d.shape}')
    print(f'new_feature shape: {new_feature[0].d.shape}')
    print(f'old_feature: {old_feature.d}')
    print(f'new_feature: {new_feature.d}')
    scriptdir = pathlib.Path(__file__).parent
    text_encoder2.save_parameters(f'{scriptdir}/data/text_encoder.h5')


def convert_old_model_to_new_model():
    scriptdir = pathlib.Path(__file__).parent
    image = Image.open(f'{scriptdir}/CLIP.png')
    text = ['hello japan', 'hello world', 'hello sony']

    import nnabla.ext_utils
    gpu_context = nnabla.ext_utils.get_extension_context('cudnn', device_id=0)
    with nn.context_scope(gpu_context):
        # load params
        load_model(f'{scriptdir}/data/RN50.h5')  # RN50 = Resnet50
        original_params = nn.get_parameters()
        image = nnabla_clip.preprocess(image)
        image = nn.Variable.from_numpy_array(image)
        convert_image_encoder_model(image, original_params)
        convert_text_encoder_model(text, original_params)


def main():
    convert_old_model_to_new_model()


if __name__ == '__main__':
    main()
