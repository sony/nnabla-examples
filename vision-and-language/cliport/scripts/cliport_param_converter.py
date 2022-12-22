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

import argparse

import nnabla as nn
import nnabla_cliport.clip.clip as nnabla_clip
from nnabla_cliport.models.clip_text_encoder import CLIPTextEncoder
from nnabla_cliport.models.clip_image_encoder import CLIPImageEncoder
from nnabla_cliport.models.cliport_attention import CLIPortAttention
from nnabla_cliport.models.cliport_transport import CLIPortTransport


def load_model(filepath):
    nn.clear_parameters()
    nn.load_parameters(filepath)


def reordered_upblock_keys(prefix, layer):
    reordered_keys = []
    reordered_keys.append(
        f'{prefix}_stream_two/up{layer}/conv/double_conv/0/W')
    reordered_keys.append(
        f'{prefix}_stream_two/up{layer}/conv/double_conv/0/b')
    reordered_keys.append(
        f'{prefix}_stream_two/up{layer}/conv/double_conv/1/b')
    reordered_keys.append(
        f'{prefix}_stream_two/up{layer}/conv/double_conv/1/W')
    reordered_keys.append(
        f'{prefix}_stream_two/up{layer}/conv/double_conv/1/mean')
    reordered_keys.append(
        f'{prefix}_stream_two/up{layer}/conv/double_conv/1/var')
    reordered_keys.append(
        f'{prefix}_stream_two/up{layer}/conv/double_conv/3/W')
    reordered_keys.append(
        f'{prefix}_stream_two/up{layer}/conv/double_conv/3/b')
    reordered_keys.append(
        f'{prefix}_stream_two/up{layer}/conv/double_conv/4/b')
    reordered_keys.append(
        f'{prefix}_stream_two/up{layer}/conv/double_conv/4/W')
    reordered_keys.append(
        f'{prefix}_stream_two/up{layer}/conv/double_conv/4/mean')
    reordered_keys.append(
        f'{prefix}_stream_two/up{layer}/conv/double_conv/4/var')
    return reordered_keys


def conv_and_identitiy_keys(prefix, layer):
    keys = []
    keys.append(f'{prefix}_stream_two/layer{layer}/0/conv1/W')
    keys.append(f'{prefix}_stream_two/layer{layer}/0/conv2/W')
    keys.append(f'{prefix}_stream_two/layer{layer}/0/conv3/W')
    keys.append(f'{prefix}_stream_two/layer{layer}/0/shortcut/0/W')
    keys.append(f'{prefix}_stream_two/layer{layer}/1/conv1/W')
    keys.append(f'{prefix}_stream_two/layer{layer}/1/conv2/W')
    keys.append(f'{prefix}_stream_two/layer{layer}/1/conv3/W')
    return keys


def reorder_stream_two_keys(prefix):
    reordered_keys = []
    reordered_keys.append(f'{prefix}_stream_two/conv1/0/W')
    reordered_keys.append(f'{prefix}_stream_two/lang_proj1/W')
    reordered_keys.append(f'{prefix}_stream_two/lang_proj1/b')
    reordered_keys.extend(reordered_upblock_keys(prefix, 1))
    reordered_keys.append(f'{prefix}_stream_two/lat_fusion1/conv/1/W')
    reordered_keys.append(f'{prefix}_stream_two/lang_proj2/W')
    reordered_keys.append(f'{prefix}_stream_two/lang_proj2/b')
    reordered_keys.extend(reordered_upblock_keys(prefix, 2))
    reordered_keys.append(f'{prefix}_stream_two/lat_fusion2/conv/1/W')
    reordered_keys.append(f'{prefix}_stream_two/lang_proj3/W')
    reordered_keys.append(f'{prefix}_stream_two/lang_proj3/b')
    reordered_keys.extend(reordered_upblock_keys(prefix, 3))
    reordered_keys.append(f'{prefix}_stream_two/lat_fusion3/conv/1/W')
    reordered_keys.extend(conv_and_identitiy_keys(prefix, 1))
    reordered_keys.append(f'{prefix}_stream_two/lat_fusion4/conv/1/W')
    reordered_keys.extend(conv_and_identitiy_keys(prefix, 2))
    reordered_keys.append(f'{prefix}_stream_two/lat_fusion5/conv/1/W')
    reordered_keys.extend(conv_and_identitiy_keys(prefix, 3))
    reordered_keys.append(f'{prefix}_stream_two/lat_fusion6/conv/1/W')
    reordered_keys.append(f'{prefix}_stream_two/conv2/0/W')
    reordered_keys.append(f'{prefix}_stream_two/conv2/0/b')

    return reordered_keys


def reorder_clip_visual_keys(keys):
    reordered_keys = []
    for key in keys:
        if ('bn' in key or 'downsample/1' in key) and 'W' in key:
            gamma_key = key
        elif ('bn' in key or 'downsample/1' in key) and 'b' in key and 'var' not in key and 'mean' not in key:
            reordered_keys.append(key)
            reordered_keys.append(gamma_key)
            print(f'reordered key: {key}')
            print(f'reordered key: {gamma_key}')
        else:
            reordered_keys.append(key)
            print(f'reordered key: {key}')
    return reordered_keys


def transfer_model_parameters(src_params, dst_params):
    for (src_key, src_param), (dst_key, dst_param) in zip(src_params.items(), dst_params.items()):
        print(f'old_param name: {src_key}')
        print(f'new_param name: {dst_key}')
        if 'affine/W' in dst_key:
            dst_param.d = np.transpose(src_param.d, axes=(1, 0))
        else:
            dst_param.d = np.reshape(src_param.d, newshape=dst_param.shape)


def convert_cliport_attention_model(image, text, original_params, args):
    scriptdir = pathlib.Path(__file__).parent
    stream_one_keys = []
    stream_two_keys = []
    clip_visual_keys = []
    for key, param in original_params.items():
        if 'num_batches_tracked' in key:
            continue
        if 'clip' in key:
            if 'visual' in key:
                clip_visual_keys.append(key)
            continue
        elif 'stream_one' in key:
            stream_one_keys.append(key)
        else:
            stream_two_keys.append(key)
        print(f'original key {key}, param {param.shape}')
    stream_two_keys = reorder_stream_two_keys('attn')
    model_keys = stream_one_keys + stream_two_keys

    image_encoder = CLIPImageEncoder('attention_image_encoder')
    text_encoder = CLIPTextEncoder('text_encoder')
    text_encoder.load_parameters(
        f'{scriptdir}/{args.param_dir}/text_encoder.h5')

    rgbd = nn.Variable.from_numpy_array(
        np.random.random(size=(1, 6, 320, 320)))
    model = CLIPortAttention('attention', image_encoder, text_encoder)
    model.compute_features(rgbd, text)

    # Attention network params
    src_params = {}
    dst_params = model.get_parameters(grad_only=False)
    for key in model_keys:
        param = original_params[key]
        src_params[key] = param
    transfer_model_parameters(src_params, dst_params)
    model.save_parameters(f'{scriptdir}/{args.param_dir}/attention.h5')

    # Image encoder params
    clip_visual_keys = reorder_clip_visual_keys(clip_visual_keys)
    src_params = {}
    dst_params = image_encoder.get_parameters(grad_only=False)
    for key in clip_visual_keys:
        param = original_params[key]
        src_params[key] = param
    transfer_model_parameters(src_params, dst_params)
    image_encoder.save_parameters(
        f'{scriptdir}/{args.param_dir}/attention_image_encoder.h5')


def convert_cliport_transport_model(image, text, original_params, args):
    scriptdir = pathlib.Path(__file__).parent
    key_stream_one_keys = []
    key_stream_two_keys = []
    key_stream_fusion_keys = []
    query_stream_one_keys = []
    query_stream_two_keys = []
    query_stream_fusion_keys = []
    clip_key_visual_keys = []
    clip_query_visual_keys = []
    for key, param in original_params.items():
        if 'num_batches_tracked' in key:
            continue
        if 'clip' in key:
            print(f'clip key: {key}')
            if 'visual' in key and 'key' in key:
                clip_key_visual_keys.append(key)
            elif 'visual' in key and 'query' in key:
                clip_query_visual_keys.append(key)
        elif 'stream_one' in key:
            if 'key_stream' in key:
                key_stream_one_keys.append(key)
            else:
                query_stream_one_keys.append(key)
        elif 'stream_two' in key:
            if 'key_stream' in key:
                key_stream_two_keys.append(key)
            else:
                query_stream_two_keys.append(key)
        else:
            if 'key' in key:
                key_stream_fusion_keys.append(key)
            else:
                query_stream_fusion_keys.append(key)
        print(f'original key {key}, param {param.shape}')
    key_stream_two_keys = reorder_stream_two_keys('key')
    query_stream_two_keys = reorder_stream_two_keys('query')
    model_keys = key_stream_one_keys + key_stream_two_keys + key_stream_fusion_keys + \
        query_stream_one_keys + query_stream_two_keys + query_stream_fusion_keys

    model_params = {}
    for key in model_keys:
        param = original_params[key]
        model_params[key] = param

    key_image_encoder = CLIPImageEncoder('transport_key_image_encoder')
    query_image_encoder = CLIPImageEncoder('transport_query_image_encoder')
    text_encoder = CLIPTextEncoder('text_encoder')
    text_encoder.load_parameters(
        f'{scriptdir}/{args.param_dir}/text_encoder.h5')

    rgbd = nn.Variable.from_numpy_array(
        np.random.random(size=(1, 6, 320, 320)))

    model = CLIPortTransport(
        'transport', key_image_encoder, query_image_encoder, text_encoder)
    pivot = (160, 160)  # dummy pivot
    model.compute_features(rgbd, text, pivot)
    new_params = model.get_parameters(grad_only=False)

    for old_param, (new_key, new_param) in zip(model_params.values(), new_params.items()):
        print(f'new_param name: {new_key}')
        print(f'old param shape: {old_param.shape}')
        print(f'new param shape: {new_param.shape}')
        if 'affine/W' in new_key:
            new_param.d = np.transpose(old_param.d, axes=(1, 0))
        else:
            new_param.d = np.reshape(old_param.d, newshape=new_param.shape)

    model.save_parameters(f'{scriptdir}/{args.param_dir}/transport.h5')

    # Image encoder params
    # key
    clip_key_visual_keys = reorder_clip_visual_keys(clip_key_visual_keys)
    src_params = {}
    dst_params = key_image_encoder.get_parameters(grad_only=False)
    for key in clip_key_visual_keys:
        param = original_params[key]
        src_params[key] = param
    transfer_model_parameters(src_params, dst_params)
    key_image_encoder.save_parameters(
        f'{scriptdir}/{args.param_dir}/transport_key_image_encoder.h5')

    # query
    clip_query_visual_keys = reorder_clip_visual_keys(clip_query_visual_keys)
    src_params = {}
    dst_params = query_image_encoder.get_parameters(grad_only=False)
    for key in clip_key_visual_keys:
        param = original_params[key]
        src_params[key] = param
    transfer_model_parameters(src_params, dst_params)
    query_image_encoder.save_parameters(
        f'{scriptdir}/{args.param_dir}/transport_query_image_encoder.h5')


def convert_cliport_model(image, text, original_params, args):
    if args.model_type == 'attention':
        convert_cliport_attention_model(image, text, original_params, args)
    elif args.model_type == 'transport':
        convert_cliport_transport_model(image, text, original_params, args)
    else:
        raise NotImplementedError


def convert_old_model_to_new_model(args):
    scriptdir = pathlib.Path(__file__).parent
    image = Image.open(f'{scriptdir}/CLIP.png')
    text = ['hello japan']

    import nnabla.ext_utils
    gpu_context = nnabla.ext_utils.get_extension_context('cudnn', device_id=0)
    with nn.context_scope(gpu_context):
        # load params
        load_model(f'{scriptdir}/cliport_{args.model_type}.h5')
        original_params = nn.get_parameters()
        image = nnabla_clip.preprocess(image)
        image = nn.Variable.from_numpy_array(image)
        token = nn.Variable.from_numpy_array(nnabla_clip.tokenize(text))
        convert_cliport_model(image, token, original_params, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--param-dir', type=str, default='data')
    parser.add_argument('--model-type', type=str,
                        choices=['attention', 'transport'], default='transport')
    args = parser.parse_args()
    convert_old_model_to_new_model(args)


if __name__ == '__main__':
    main()
