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

import numpy
import zipfile

import nnabla as nn
import tensrflow as tf
import nnabla.parametric_functions as PF

from subprocess import call
from nnabla.utils.data_source_loader import download, get_data_home
from tensorflow.python import pywrap_tensorflow


def convert(ckpt_file, h5_file):
    """
    Convert BERT Tensorflow weights to NNabla

    Args:
        ckpt_file: Input Tensorflow ckpt file
        h5_file: Output NNabla output file 

    """
    # Check the TensorFlow version for compatibility
    if int(tf.__version__[0]) == 2:
        reader = tf.compat.v1.train.NewCheckpointReader(ckpt_file)
    else:
        reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)
    var_to_shape_map = reader.get_variable_to_shape_map()

    for key in sorted(var_to_shape_map):
        weight = reader.get_tensor(key)
        if 'encoder' in key:
            layer_id = int(key.split('/')[2].replace('layer_', ''))
            if 'query/bias' in key:
                key = 'encoder{:02d}/transformer_encode/src_self_attn/multi_head_attention/q_bias'.format(
                    layer_id)
            if 'query/kernel' in key:
                key = 'encoder{:02d}/transformer_encode/src_self_attn/multi_head_attention/q_weight'.format(
                    layer_id)
            if 'key/bias' in key:
                key = 'encoder{:02d}/transformer_encode/src_self_attn/multi_head_attention/k_bias'.format(
                    layer_id)
            if 'key/kernel' in key:
                key = 'encoder{:02d}/transformer_encode/src_self_attn/multi_head_attention/k_weight'.format(
                    layer_id)
            if 'value/bias' in key:
                key = 'encoder{:02d}/transformer_encode/src_self_attn/multi_head_attention/v_bias'.format(
                    layer_id)
            if 'value/kernel' in key:
                key = 'encoder{:02d}/transformer_encode/src_self_attn/multi_head_attention/v_weight'.format(
                    layer_id)
            if 'attention/output/LayerNorm/beta' in key:
                key = 'encoder{:02d}/transformer_encode/enc_layer_norm1/layer_normalization/beta'.format(
                    layer_id)
                weight = numpy.reshape(weight, (1, 1, 768))
            elif 'output/LayerNorm/beta' in key:
                key = 'encoder{:02d}/transformer_encode/enc_layer_norm2/layer_normalization/beta'.format(
                    layer_id)
                weight = numpy.reshape(weight, (1, 1, 768))
            if 'attention/output/LayerNorm/gamma' in key:
                key = 'encoder{:02d}/transformer_encode/enc_layer_norm1/layer_normalization/gamma'.format(
                    layer_id)
                weight = numpy.reshape(weight, (1, 1, 768))
            elif 'output/LayerNorm/gamma' in key:
                key = 'encoder{:02d}/transformer_encode/enc_layer_norm2/layer_normalization/gamma'.format(
                    layer_id)
                weight = numpy.reshape(weight, (1, 1, 768))
            if 'attention/output/dense/bias' in key:
                key = 'encoder{:02d}/transformer_encode/src_self_attn/multi_head_attention/out_bias'.format(
                    layer_id)
            elif 'output/dense/bias' in key:
                key = 'encoder{:02d}/transformer_encode/enc_affine2/affine/b'.format(
                    layer_id)
            if 'intermediate/dense/bias' in key:
                key = 'encoder{:02d}/transformer_encode/enc_affine1/affine/b'.format(
                    layer_id)
            if 'attention/output/dense/kernel' in key:
                key = 'encoder{:02d}/transformer_encode/src_self_attn/multi_head_attention/out_weight'.format(
                    layer_id)
            elif 'output/dense/kernel' in key:
                key = 'encoder{:02d}/transformer_encode/enc_affine2/affine/W'.format(
                    layer_id)
            if 'intermediate/dense/kernel' in key:
                key = 'encoder{:02d}/transformer_encode/enc_affine1/affine/W'.format(
                    layer_id)
        if 'embeddings/LayerNorm/' in key:
            key = key.replace('bert/embeddings/LayerNorm',
                              'embed/layer_normalization')
            weight = numpy.reshape(weight, (1, 1, 768))
        if 'word_embeddings' in key:
            key = 'word_embeddings/embed/W'
        if 'token_type_embeddings' in key:
            key = 'token_type_embeddings/embed/W'
        if 'position_embeddings' in key:
            key = 'position_embeddings/embed/W'
        if 'pooler/dense/bias' in key:
            key = 'pooler/affine/b'
        if 'pooler/dense/kernel' in key:
            key = 'pooler/affine/W'
        if 'seq_relationship/output_weights' in key:
            key = 'affine_seq_class/affine/W'
            weight = numpy.transpose(weight)
        if 'seq_relationship/output_bias' in key:
            key = 'affine_seq_class/affine/b'

        params = PF.get_parameter_or_create(key, shape=weight.shape)
        params.d = weight

    nn.parameter.save_parameters(h5_file)


def main():

    output_nnabla_file = 'nbla_bert_params.h5'
    r = download(
        "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip", 'uncased_L-12_H-768_A-12.zip')

    with zipfile.ZipFile("uncased_L-12_H-768_A-12.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    input_ckpt_file = 'uncased_L-12_H-768_A-12/bert_model.ckpt'
    output_nnabla_file = 'nbla_bert_params.h5'
    convert(input_ckpt_file, output_nnabla_file)


if __name__ == '__main__':
    main()
