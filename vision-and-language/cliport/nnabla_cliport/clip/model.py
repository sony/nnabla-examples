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

import re

import nnabla as nn
import nnabla.functions as F


def _in_projection_packed(q, k, v, w, b):
    if k is v:
        if q is k:
            # self-attention
            w = F.transpose(w, (1, 0))
            to_ret = F.affine(q, w, b, base_axis=2)
            ind = -(-to_ret.size_from_axis(2) // 3)
            a, b, c = to_ret.shape
            return F.slice(to_ret, (0, 0, 0), (a, b, ind)), F.slice(to_ret, (0, 0, ind), (a, b, ind*2)), F.slice(to_ret, (0, 0, ind*2), (a, b, c))
        else:
            # encoder-decoder attention
            raise NotImplementedError()
    else:
        raise NotImplementedError()


def _in_projection(q, k, v, w_q, w_k, w_v, b_q, b_k, b_v):
    Eq, Ek, Ev = q.shape[-1], k.shape[-1], v.shape[-1]
    assert w_q.shape == (
        Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (
        Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (
        Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (
        Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (
        Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (
        Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"

    return F.affine(q, w_q, b_q, base_axis=2), F.affine(k, w_k, b_k, base_axis=2), F.affine(v, w_v, b_v, base_axis=2)


def _scaled_dot_product_attention(q, k, v, attn_mask, dropout):
    B, Nt, E = q.shape
    q *= float(E) ** -0.5
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = F.batch_matmul(q, k, transpose_b=True)
    if attn_mask is not None:
        attn += attn_mask
    attn_output_weights = F.softmax(attn, axis=len(attn.shape)-1)
    if dropout > 0.0:
        attn = F.dropout(attn, p=dropout)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    attn_output = F.batch_matmul(attn_output_weights, v)
    return attn_output, attn_output_weights


def multi_head_attention(query, key, value, d_model, num_heads, need_weights=False, attn_mask=None,
                         use_separate_proj_weight=False,
                         q_proj_weight=None,
                         k_proj_weight=None,
                         v_proj_weight=None,
                         in_proj_bias=None,
                         out_proj_weight=None,
                         out_proj_bias=None):
    tgt_len, batch_size, embed_dim = query.shape
    src_len, _, _ = key.shape

    head_dim = d_model // num_heads
    assert head_dim * num_heads == embed_dim, 'embed_dim must be divisible by num_heads'

    if attn_mask is not None:
        if attn_mask.ndim == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.reshape((1, tgt_len, src_len))
        elif attn_mask.dim() == 3:
            correct_3d_size = (batch_size * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(
                f"attn_mask's dimension {attn_mask.ndim} is not supported")
    if not use_separate_proj_weight:
        in_proj_weight = nn.parameter.get_parameter_or_create(
            name="attn/in_proj_W", shape=(d_model*3, d_model))
        in_proj_bias = nn.parameter.get_parameter_or_create(
            name="attn/in_proj_b", shape=(d_model*3,))
        q, k, v = _in_projection_packed(
            query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"

        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias
        q_proj_weight = F.transpose(q_proj_weight, (1, 0))
        k_proj_weight = F.transpose(k_proj_weight, (1, 0))
        v_proj_weight = F.transpose(v_proj_weight, (1, 0))
        q, k, v = _in_projection(
            query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    q = F.transpose(
        F.reshape(q, (tgt_len, batch_size * num_heads, head_dim)), (1, 0, 2))  # q:(B*H, L_T, head_dim)
    k = F.transpose(
        F.reshape(k, (-1, batch_size * num_heads, head_dim)), (1, 0, 2))  # k:(B*H, L_S, head_dim)
    v = F.transpose(
        F.reshape(v, (-1, batch_size * num_heads, head_dim)), (1, 0, 2))  # v:(B*H, L_S, head_vdim)

    dropout_p = 0.0

    attn_output, attn_output_weights = _scaled_dot_product_attention(
        q, k, v, attn_mask, dropout_p)
    attn_output = F.reshape(F.transpose(
        attn_output, (1, 0, 2)), (tgt_len, batch_size, embed_dim))  # attn_output: (L_T, B, E_v)

    if out_proj_weight is None:
        out_proj_weight = nn.parameter.get_parameter_or_create(
            name="attn/out_proj/W", shape=(d_model, d_model))
    if out_proj_bias is None:
        out_proj_bias = nn.parameter.get_parameter_or_create(
            name="attn/out_proj/b", shape=(d_model,))

    out_proj_weight = F.transpose(out_proj_weight, (1, 0))
    attn_output = F.affine(attn_output, out_proj_weight,
                           out_proj_bias, base_axis=2)

    return attn_output


def layernorm(x, i, d_model):
    weight = nn.parameter.get_parameter_or_create(
        name=f"ln_{i}/W", shape=(d_model,)).reshape((1, 1, d_model))
    bias = nn.parameter.get_parameter_or_create(
        name=f"ln_{i}/b", shape=(d_model,)).reshape((1, 1, d_model))

    return F.layer_normalization(x, bias, weight, batch_axis=(0, 1))


def quick_gelu(x):
    return x * F.sigmoid(1.702 * x)


def mlp(x):
    _, _, dim = x.shape
    c_fc_w = nn.parameter.get_parameter_or_create(
        name="mlp/c_fc/W", shape=(dim*4, dim))
    c_fc_b = nn.parameter.get_parameter_or_create(
        name="mlp/c_fc/b", shape=(dim*4,))
    c_proj_w = nn.parameter.get_parameter_or_create(
        name="mlp/c_proj/W", shape=(dim, dim*4))
    c_proj_b = nn.parameter.get_parameter_or_create(
        name="mlp/c_proj/b", shape=(dim,))

    c_fc_w = F.transpose(c_fc_w, (1, 0))
    h1 = F.affine(x, c_fc_w, c_fc_b, base_axis=2)
    h2 = quick_gelu(h1)
    c_proj_w = F.transpose(c_proj_w, (1, 0))
    proj_out = F.affine(h2, c_proj_w, c_proj_b, base_axis=2)

    return proj_out


def residual_attention_block(x, d_model, num_heads, res_num, attn_mask=None):
    with nn.parameter_scope(f"resblocks/{res_num}"):
        x_norm1 = layernorm(x, 1, d_model)
        x = x + multi_head_attention(
            x_norm1, x_norm1, x_norm1, d_model, num_heads,
            need_weights=False, attn_mask=attn_mask)
        x_norm2 = layernorm(x, 2, d_model)
        x = x + mlp(x_norm2)

    return x


def transformer(x, width, layers, heads, attn_mask=None):
    with nn.parameter_scope("transformer"):
        for i in range(layers):
            x = residual_attention_block(x, width, heads, i, attn_mask)

    return x


def vision_transformer(x, input_res, patch_size, v_width, v_layers, v_heads, embed_dim):
    with nn.parameter_scope("visual"):
        con1_w = nn.parameter.get_parameter_or_create(
            name="conv1/W", shape=(v_width, 3, patch_size, patch_size))
        x = F.convolution(x, con1_w, bias=None, stride=(
            patch_size, patch_size))  # shape = [*, width, grid, grid]

        # shape = [*, width, grid ** 2]
        x = F.reshape(x, (x.shape[0], x.shape[1], -1))
        x = F.transpose(x, (0, 2, 1))  # shape = [*, grid ** 2, width]

        z = np.zeros((x.shape[0], 1, x.shape[-1]))
        zeros = nn.Variable.from_numpy_array(z)
        class_embed = nn.parameter.get_parameter_or_create(
            name="class_embedding", shape=(v_width,)).reshape((x.shape[0], 1, v_width))
        # shape = [*, grid ** 2 + 1, width]
        x = F.concatenate(class_embed + zeros, x, axis=1)

        positional_embedding = nn.parameter.get_parameter_or_create(
            name='positional_embedding', shape=((input_res // patch_size) ** 2 + 1, v_width)).reshape((x.shape[0], x.shape[1], v_width))
        x = x + positional_embedding

        ln_pre_w = nn.parameter.get_parameter_or_create(
            name="ln_pre/W", shape=(v_width,)).reshape((1, 1, v_width))
        ln_pre_b = nn.parameter.get_parameter_or_create(
            name="ln_pre/b", shape=(v_width,)).reshape((1, 1, v_width))
        x = F.layer_normalization(x, ln_pre_b, ln_pre_w, batch_axis=(0, 1))

        x = F.transpose(x, (1, 0, 2))  # NLD -> LND

        x = transformer(x, v_width, v_layers, v_heads)

        x = F.transpose(x, (1, 0, 2))  # LND -> NLD

        ln_post_w = nn.parameter.get_parameter_or_create(
            name="ln_post/W", shape=(v_width,)).reshape((1, 1, v_width))
        ln_post_b = nn.parameter.get_parameter_or_create(
            name="ln_post/b", shape=(v_width,)).reshape((1, 1, v_width))
        x = F.slice(x, stop=(x.shape[0], 1, x.shape[2]))
        x = F.layer_normalization(x, ln_post_b, ln_post_w)

        if 'proj' in nn.get_parameters():
            visual_proj = nn.parameter.get_parameter_or_create(
                name="proj", shape=(v_width, embed_dim)).reshape((1, v_width, -1))
            x = F.batch_matmul(x, visual_proj)

        x = x.reshape((-1, embed_dim))

    return x


def batch_normalization(inp, gamma, beta, mean, var):
    gamma = F.reshape(gamma, shape=(1, *gamma.shape, 1, 1))
    beta = F.reshape(beta, shape=(1, *beta.shape, 1, 1))
    mean = F.reshape(mean, shape=(1, *mean.shape, 1, 1))
    var = F.reshape(var, shape=(1, *var.shape, 1, 1))
    return F.batch_normalization(inp, gamma=gamma, beta=beta, mean=mean, variance=var, batch_stat=False)


def stem(x, v_width):
    conv1_w = nn.parameter.get_parameter("conv1/W")
    assert conv1_w.shape == (v_width // 2, 3, 3, 3)
    x = F.convolution(x, conv1_w, bias=None, stride=(2, 2), pad=(1, 1))

    with nn.parameter_scope('bn1'):
        bn1_b = nn.parameter.get_parameter("b")
        bn1_w = nn.parameter.get_parameter("W")
        mean = nn.parameter.get_parameter("mean")
        var = nn.parameter.get_parameter("var")
        assert bn1_w.shape == (v_width // 2,)
        assert bn1_b.shape == (v_width // 2,)
        x = batch_normalization(x, gamma=bn1_w, beta=bn1_b, mean=mean, var=var)
        x = F.relu(x)

    conv2_w = nn.parameter.get_parameter("conv2/W")
    assert conv2_w.shape == (v_width // 2, v_width // 2, 3, 3)
    x = F.convolution(x, conv2_w, bias=None, pad=(1, 1))

    with nn.parameter_scope('bn2'):
        bn2_b = nn.parameter.get_parameter("b")
        bn2_w = nn.parameter.get_parameter("W")
        mean = nn.parameter.get_parameter("mean")
        var = nn.parameter.get_parameter("var")
        assert bn2_w.shape == (v_width // 2,)
        assert bn2_b.shape == (v_width // 2,)
        x = batch_normalization(x, gamma=bn2_w, beta=bn2_b, mean=mean, var=var)
        x = F.relu(x)

    conv3_w = nn.parameter.get_parameter("conv3/W")
    assert conv3_w.shape == (v_width, v_width // 2, 3, 3)
    x = F.convolution(x, conv3_w, bias=None, pad=(1, 1))

    with nn.parameter_scope('bn3'):
        bn3_b = nn.parameter.get_parameter("b")
        bn3_w = nn.parameter.get_parameter("W")
        mean = nn.parameter.get_parameter("mean")
        var = nn.parameter.get_parameter("var")
        assert bn3_w.shape == (v_width,)
        assert bn3_b.shape == (v_width,)
        x = batch_normalization(x, gamma=bn3_w, beta=bn3_b, mean=mean, var=var)
        x = F.relu(x)
    return F.average_pooling(x, kernel=(2, 2))


def bottleneck(x, inplanes, planes, stride=1, expansion=4):
    identity = x

    conv1_w = nn.parameter.get_parameter("conv1/W")
    assert conv1_w.shape == (planes, inplanes, 1, 1)
    out = F.convolution(x, conv1_w, bias=None, stride=(1, 1))
    bn1_b = nn.parameter.get_parameter("bn1/b")
    bn1_w = nn.parameter.get_parameter("bn1/W")
    mean = nn.parameter.get_parameter("bn1/mean")
    var = nn.parameter.get_parameter("bn1/var")
    assert bn1_w.shape == (planes,)
    assert bn1_b.shape == (planes,)
    with nn.parameter_scope('bn1'):
        out = batch_normalization(
            out, beta=bn1_b, gamma=bn1_w, mean=mean, var=var)
    out = F.relu(out)

    conv2_w = nn.parameter.get_parameter("conv2/W")
    assert conv2_w.shape == (planes, planes, 3, 3)
    out = F.convolution(out, conv2_w, bias=None, stride=(1, 1), pad=(1, 1))
    bn2_b = nn.parameter.get_parameter("bn2/b")
    bn2_w = nn.parameter.get_parameter("bn2/W")
    mean = nn.parameter.get_parameter("bn2/mean")
    var = nn.parameter.get_parameter("bn2/var")
    assert bn2_w.shape == (planes,)
    assert bn2_b.shape == (planes,)
    with nn.parameter_scope('bn2'):
        out = batch_normalization(
            out, beta=bn2_b, gamma=bn2_w, mean=mean, var=var)
    out = F.relu(out)

    if stride > 1:
        out = F.average_pooling(out, kernel=(stride, stride))

    conv3_w = nn.parameter.get_parameter("conv3/W")
    assert conv3_w.shape == (planes * expansion, planes, 1, 1)
    out = F.convolution(out, conv3_w, bias=None, stride=(1, 1))
    bn3_b = nn.parameter.get_parameter("bn3/b")
    bn3_w = nn.parameter.get_parameter("bn3/W")
    mean = nn.parameter.get_parameter("bn3/mean")
    var = nn.parameter.get_parameter("bn3/var")
    assert bn3_w.shape == (planes * expansion,)
    assert bn3_b.shape == (planes * expansion,)
    with nn.parameter_scope('bn3'):
        out = batch_normalization(
            out, beta=bn3_b, gamma=bn3_w, mean=mean, var=var)

    if stride > 1 or inplanes != planes * expansion:
        # downsample
        # NOTE: stride size is used as kernel size in original code
        downsample = F.average_pooling(x, kernel=(stride, stride))

        conv4_w = nn.parameter.get_parameter("downsample/0/W")
        assert conv4_w.shape == (planes * expansion, inplanes, 1, 1)
        downsample = F.convolution(
            downsample, conv4_w, bias=None, stride=(1, 1))

        bn4_b = nn.parameter.get_parameter("downsample/1/b")
        bn4_w = nn.parameter.get_parameter("downsample/1/W")
        mean = nn.parameter.get_parameter("downsample/1/mean")
        var = nn.parameter.get_parameter("downsample/1/var")
        assert bn4_w.shape == (planes * expansion,)
        assert bn4_b.shape == (planes * expansion,)
        with nn.parameter_scope('bn4'):
            downsample = batch_normalization(
                downsample, beta=bn4_b, gamma=bn4_w, mean=mean, var=var)

        identity = downsample

    out += identity
    return F.relu(out)


def resnet_layer(x, blocks, planes, inplanes, stride=1, expansion=4):
    for i in range(0, blocks):
        if i != 0:
            inplanes = planes * expansion
            stride = 1
        with nn.parameter_scope(f'{i}'):
            x = bottleneck(x, inplanes, planes, stride, expansion)
    return x, inplanes


def attention_pool_2d(x, spacial_dim, embed_dim, num_heads, output_dim=None):
    x = x.reshape(shape=(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
    x = F.transpose(x, axes=(2, 0, 1))  # NCHW -> (HW)NC
    x_mean = F.mean(x, axis=0, keepdims=True)
    x = F.concatenate(x_mean, x, axis=0)  # (HW+1)NC

    positional_embedding = nn.parameter.get_parameter('positional_embedding')
    assert positional_embedding.shape == (spacial_dim ** 2 + 1, embed_dim)
    positional_embedding = positional_embedding.reshape(
        (x.shape[0], x.shape[1], embed_dim))

    x = x + positional_embedding

    q_proj_weight = nn.parameter.get_parameter("q_proj/W")
    q_proj_bias = nn.parameter.get_parameter("q_proj/b")
    k_proj_weight = nn.parameter.get_parameter("k_proj/W")
    k_proj_bias = nn.parameter.get_parameter("k_proj/b")
    v_proj_weight = nn.parameter.get_parameter("v_proj/W")
    v_proj_bias = nn.parameter.get_parameter("v_proj/b")
    out_proj_weight = nn.parameter.get_parameter("c_proj/W")
    out_proj_bias = nn.parameter.get_parameter("c_proj/b")

    assert q_proj_weight.shape == (embed_dim, embed_dim)
    assert q_proj_bias.shape == (embed_dim,)
    assert k_proj_weight.shape == (embed_dim, embed_dim)
    assert k_proj_bias.shape == (embed_dim,)
    assert v_proj_weight.shape == (embed_dim, embed_dim)
    assert v_proj_bias.shape == (embed_dim,)
    assert out_proj_weight.shape == (output_dim or embed_dim, embed_dim)
    assert out_proj_bias.shape == (output_dim or embed_dim,)

    x = multi_head_attention(x, x, x,
                             d_model=x.shape[-1],
                             num_heads=num_heads,
                             q_proj_weight=q_proj_weight,
                             k_proj_weight=k_proj_weight,
                             v_proj_weight=v_proj_weight,
                             in_proj_bias=(
                                 q_proj_bias, k_proj_bias, v_proj_bias),
                             out_proj_weight=out_proj_weight,
                             out_proj_bias=out_proj_bias,
                             use_separate_proj_weight=True)

    x = F.split(x, axis=0)[0]
    return x


def prepool(x, layers, width=64):
    mid_features = []
    x = stem(x, width)
    inplanes = width
    for i, blocks in enumerate(layers):
        with nn.parameter_scope(f'layer{i+1}'):
            stride = 1 if i == 0 else 2
            x, inplanes = resnet_layer(
                x, blocks, width * (2**i), inplanes, stride=stride)
            mid_features.append(x)

    return x, mid_features


def modified_resnet(x, layers, output_dim, heads, input_resolution=224, width=64):
    with nn.parameter_scope("visual"):
        x, mid_features = prepool(x, layers, width=width)

        embed_dim = width * 32  # the ResNet feature dimension

        with nn.parameter_scope('attnpool'):
            x = attention_pool_2d(x, input_resolution //
                                  32, embed_dim, heads, output_dim)

    return x, mid_features


def modified_resnet_no_pool(x, layers, width=64):
    with nn.parameter_scope("visual"):
        x, mid_features = prepool(x, layers, width=width)
    return x, mid_features


def build_attn_mask(context_len):
    mask = np.empty((context_len, context_len))
    mask.fill(float('-inf'))
    mask = np.triu(mask, 1)

    return nn.Variable.from_numpy_array(mask)


def logits(image, text):
    image_features = encode_image(image)
    text_features = encode_text(text)

    # normalized features
    image_features = image_features / \
        F.norm(image_features, axis=1, keepdims=True)
    text_features = text_features / \
        F.norm(text_features, axis=1, keepdims=True)

    # cosine similarity as logits
    logit_scale = nn.parameter.get_parameter_or_create(
        name='logit_scale', shape=())
    logit_scale = F.exp(logit_scale)

    image_features = image_features.reshape(
        (1, image_features.shape[0], image_features.shape[1]))
    text_features = F.transpose(text_features, (1, 0))
    text_features = text_features.reshape(
        (1, text_features.shape[0], text_features.shape[1]))

    per_image = F.batch_matmul(image_features, text_features).reshape(
        (image_features.shape[0], -1))
    logits_per_image = logit_scale.reshape((1, 1)) * per_image

    logits_per_text = F.transpose(logits_per_image, (1, 0))

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text


class CLIP():
    def __init__(self) -> None:
        self.text_enc = transformer

    def encode(self, x):
        # t_e = self.text_enc(x)
        pass


def encode_image(x):
    param_dict = nn.get_parameters()

    embed_dim = param_dict['text_projection'].shape[1]

    if 'visual/proj' in param_dict:
        # use ViT
        vision_width = param_dict['visual/conv1/W'].shape[0]
        vision_layers = len([k for k in param_dict.keys() if k.startswith(
            'visual/') and k.endswith('/attn/in_proj_W')])
        vision_patch_size = param_dict['visual/conv1/W'].shape[-1]
        grid_size = round(
            (param_dict['visual/positional_embedding'].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
        vision_heads = vision_width // 64

        if not isinstance(x, nn.Variable):
            x = nn.Variable.from_numpy_array(x)

        return vision_transformer(x, image_resolution, vision_patch_size, vision_width, vision_layers, vision_heads, embed_dim)
    else:
        # Use ModifiedResNet
        vision_width = param_dict['visual/conv1/W'].shape[0] * 2
        vision_patch_size = param_dict['visual/conv1/W'].shape[-1]
        vision_layer_names = [k for k in param_dict.keys() if k.startswith(
            'visual/layer') and k.endswith('/conv1/W')]
        vision_layers = {}
        for vision_layer_name in vision_layer_names:
            match = re.search(r'layer([0-9]*).([0-9]*).*', vision_layer_name)
            layer_number = match.group(1)
            block_number = int(match.group(2)) + 1
            if layer_number in vision_layers:
                if vision_layers[layer_number] < block_number:
                    vision_layers[layer_number] = block_number
            else:
                vision_layers[layer_number] = block_number
        vision_layers = vision_layers.values()
        image_resolution = round(
            (param_dict['visual/attnpool/positional_embedding'].shape[0] - 1) ** 0.5 * 32)
        vision_heads = vision_width * 32 // 64

        if not isinstance(x, nn.Variable):
            x = nn.Variable.from_numpy_array(x)

        return modified_resnet(x, layers=vision_layers, output_dim=embed_dim, heads=vision_heads, input_resolution=image_resolution, width=vision_width)


def encode_text(text):
    param_dict = nn.get_parameters()

    embed_dim = param_dict['text_projection'].shape[1]
    context_length = param_dict['positional_embedding'].shape[0]
    vocab_size = param_dict['token_embedding/W'].shape[0]
    transformer_width = param_dict['ln_final/W'].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(
        '/')[2] for k in param_dict.keys() if k.startswith('transformer/resblocks')))

    token_embedding = nn.parameter.get_parameter_or_create(
        name='token_embedding/W', shape=(vocab_size, transformer_width))
    x = F.embed(text, token_embedding)  # [batch_size, n_ctx, d_model]

    positional_embedding = nn.parameter.get_parameter_or_create(
        name='positional_embedding', shape=(context_length, transformer_width)).reshape((1, context_length, transformer_width))
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

    idx = F.max(text, axis=-1, only_index=True)
    idx.forward()
    x = x[list(range(x.shape[0])), idx.d].reshape((1, x.shape[0], -1))
    text_projection = nn.parameter.get_parameter_or_create(
        name='text_projection', shape=(transformer_width, embed_dim)).reshape((1, transformer_width, embed_dim))
    x = F.batch_matmul(x, text_projection)

    x = x.reshape((-1, embed_dim))

    return x
