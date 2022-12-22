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
import nnabla.parametric_functions as PF

import numpy as np

import nnabla_cliport.parametric_functions as CPF
from nnabla_cliport.clip.model import transformer, build_attn_mask
from nnabla_cliport.models.model import Model


class CLIPTextEncoder(Model):
    def __init__(self,
                 scope_name,
                 embed_dim=1024,
                 vocab_size=49408,
                 transformer_layers=12, transformer_heads=8, transformer_width=512,
                 context_length=77):
        super().__init__(scope_name)
        self._embed_dim = embed_dim
        self._vocab_size = vocab_size
        self._transformer_width = transformer_width
        self._transformer_layers = transformer_layers
        self._transformer_heads = transformer_heads
        self._context_length = context_length

    def encode_text(self, text):
        with nn.parameter_scope(self._scope_name):
            with nn.parameter_scope('token_embedding'):
                x = PF.embed(text, self._vocab_size, self._transformer_width)
            with nn.parameter_scope('positional_embedding'):
                embedding_shape = (self._context_length,
                                   self._transformer_width)
                x = CPF.positional_embedding(x, embedding_shape)
            with nn.parameter_scope('transformer'):
                x = F.transpose(x, (1, 0, 2))  # NLD -> LND
                x = CPF.transformer(x,
                                    self._transformer_width,
                                    self._transformer_layers,
                                    self._transformer_heads,
                                    attn_mask=self._build_attn_mask(self._context_length))
                x = F.transpose(x, (1, 0, 2))  # LND -> NLD
            with nn.parameter_scope('layer_norm'):
                x = PF.layer_normalization(
                    x, batch_axis=(0, 1), fix_parameters=True)

            # idx = np.argmax(text.d, axis=-1)
            # x = x[list(range(x.shape[0])), idx].reshape((1, x.shape[0], -1))
            idx = F.max(text, axis=-1, keepdims=True, only_index=True)
            idx_mask = F.one_hot(idx, shape=(x.shape[1],)).reshape(
                (x.shape[0], x.shape[1], 1))
            x = F.sum(x * idx_mask, axis=1).reshape((1, x.shape[0], -1))

            with nn.parameter_scope('text_projection'):
                x = CPF.text_projection(
                    x, self._transformer_width, self._embed_dim)
            x = x.reshape((-1, self._embed_dim))
            x.need_grad = False  # Prevent updating parameters
        return x

    def _build_attn_mask(self, context_len):
        mask = np.empty((context_len, context_len))
        mask.fill(float('-inf'))
        mask = np.triu(mask, 1)

        return nn.Variable.from_numpy_array(mask)


class CLIPTextEncoderOld(object):
    def __init__(self):
        pass

    def encode_text(self, text):
        param_dict = nn.get_parameters()

        embed_dim = param_dict['text_projection'].shape[1]
        context_length = param_dict['positional_embedding'].shape[0]
        vocab_size = param_dict['token_embedding/W'].shape[0]
        transformer_width = param_dict['ln_final/W'].shape[0]
        print(f'vocab_size: {vocab_size}')
        print(f'transformer_width: {transformer_width}')
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split('/')[2]
                                     for k in param_dict.keys() if k.startswith('transformer/resblocks')))
        print(f'text embed dim: {embed_dim}')
        print(f'text context_length: {context_length}')
        print(f'text vocab_size: {vocab_size}')
        print(f'text transformer_width: {transformer_width}')
        print(f'text transformer_heads: {transformer_heads}')
        print(f'text transformer_layers: {transformer_layers}')
        token_embedding = nn.parameter.get_parameter_or_create(name='token_embedding/W',
                                                               shape=(vocab_size, transformer_width))
        x = F.embed(text, token_embedding)  # [batch_size, n_ctx, d_model]

        positional_embedding = nn.parameter.get_parameter_or_create(name='positional_embedding',
                                                                    shape=(context_length, transformer_width))
        positional_embedding = positional_embedding.reshape(
            (1, context_length, transformer_width))
        x = x + positional_embedding

        x = F.transpose(x, (1, 0, 2))  # NLD -> LND

        x = transformer(
            x, transformer_width, transformer_layers, transformer_heads, attn_mask=build_attn_mask(context_length))

        x = F.transpose(x, (1, 0, 2))  # LND -> NLD

        ln_final_W = nn.parameter.get_parameter_or_create(
            name='ln_final/W', shape=(transformer_width,)).reshape((1, 1, transformer_width))
        ln_final_b = nn.parameter.get_parameter_or_create(
            name='ln_final/b', shape=(transformer_width,)).reshape((1, 1, transformer_width))
        x = F.layer_normalization(x, ln_final_b, ln_final_W, batch_axis=(0, 1))

        idx = np.argmax(text.d, axis=-1)
        x = x[list(range(x.shape[0])), idx].reshape((1, x.shape[0], -1))

        text_projection = nn.parameter.get_parameter_or_create(name='text_projection',
                                                               shape=(transformer_width, embed_dim))
        text_projection = text_projection.reshape(
            (1, transformer_width, embed_dim))
        x = F.batch_matmul(x, text_projection)

        x = x.reshape((-1, embed_dim))
        x.need_grad = False  # Prevent updating parameters

        return x
