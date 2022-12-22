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

import nnabla_cliport.initializers as I


def cliport_spatial(rgbd, output_size, outdim):
    mid_features = []
    # spatial head
    with nn.parameter_scope('spatial_conv1'):
        spatial = PF.convolution(rgbd, outmaps=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                 w_init=I.PytorchConv2dWeightInitializer(
                                     inmaps=rgbd.shape[1], outmaps=64, kernel=(3, 3)),
                                 b_init=I.PytorchConv2dBiasInitializer(inmaps=rgbd.shape[1], outmaps=64, kernel=(3, 3)))
        spatial = F.relu(spatial)
    with nn.parameter_scope('conv_and_identity1'):
        spatial = F.relu(_conv_block(
            spatial, outmaps=[64, 64, 64], stride=(1, 1)))
        spatial = F.relu(_identity_block(
            spatial, outmaps=[64, 64, 64], stride=(1, 1)))
    with nn.parameter_scope('conv_and_identity2'):
        spatial = F.relu(_conv_block(spatial, outmaps=[
                         128, 128, 128], stride=(2, 2)))
        spatial = F.relu(_identity_block(
            spatial, outmaps=[128, 128, 128], stride=(1, 1)))
    with nn.parameter_scope('conv_and_identity3'):
        spatial = F.relu(_conv_block(spatial, outmaps=[
                         256, 256, 256], stride=(2, 2)))
        spatial = F.relu(_identity_block(
            spatial, outmaps=[256, 256, 256], stride=(1, 1)))
    with nn.parameter_scope('conv_and_identity4'):
        spatial = F.relu(_conv_block(spatial, outmaps=[
                         512, 512, 512], stride=(2, 2)))
        spatial = F.relu(_identity_block(
            spatial, outmaps=[512, 512, 512], stride=(1, 1)))
    with nn.parameter_scope('conv_and_identity5'):
        spatial = F.relu(_conv_block(spatial, outmaps=[
                         1024, 1024, 1024], stride=(2, 2)))
        spatial = F.relu(_identity_block(spatial, outmaps=[
                         1024, 1024, 1024], stride=(1, 1)))

    # spatial core
    mid_features.append(spatial)
    with nn.parameter_scope('conv_and_identity6'):
        spatial = F.relu(_conv_block(spatial, outmaps=[
                         512, 512, 512], stride=(1, 1)))
        spatial = F.relu(_identity_block(
            spatial, outmaps=[512, 512, 512], stride=(1, 1)))
        spatial = _upsample(spatial, scale=2.0)
    mid_features.append(spatial)
    with nn.parameter_scope('conv_and_identity7'):
        spatial = F.relu(_conv_block(spatial, outmaps=[
                         256, 256, 256], stride=(1, 1)))
        spatial = F.relu(_identity_block(
            spatial, outmaps=[256, 256, 256], stride=(1, 1)))
        spatial = _upsample(spatial, scale=2.0)
    mid_features.append(spatial)
    with nn.parameter_scope('conv_and_identity8'):
        spatial = F.relu(_conv_block(spatial, outmaps=[
                         128, 128, 128], stride=(1, 1)))
        spatial = F.relu(_identity_block(
            spatial, outmaps=[128, 128, 128], stride=(1, 1)))
        spatial = _upsample(spatial, scale=2.0)
    mid_features.append(spatial)
    with nn.parameter_scope('conv_and_identity9'):
        spatial = F.relu(_conv_block(
            spatial, outmaps=[64, 64, 64], stride=(1, 1)))
        spatial = F.relu(_identity_block(
            spatial, outmaps=[64, 64, 64], stride=(1, 1)))
        spatial = _upsample(spatial, scale=2.0)
    mid_features.append(spatial)
    with nn.parameter_scope('conv_and_identity10'):
        spatial = F.relu(_conv_block(
            spatial, outmaps=[32, 32, 32], stride=(1, 1)))
        spatial = F.relu(_identity_block(
            spatial, outmaps=[32, 32, 32], stride=(1, 1)))
        spatial = _upsample(spatial, scale=2.0)
    mid_features.append(spatial)
    with nn.parameter_scope('conv_and_identity11'):
        spatial = _conv_block(spatial, outmaps=[16, 16, outdim], stride=(1, 1))
        spatial = _identity_block(
            spatial, outmaps=[16, 16, outdim], stride=(1, 1))
    spatial = _downsample(spatial, output_size=output_size)

    return spatial, mid_features


def cliport_semantic(spatial, encoded_image, image_features, encoded_text, output_size, outdim, training):
    with nn.parameter_scope('semantic_conv1'):
        semantic = PF.convolution(encoded_image, outmaps=1024, kernel=(3, 3),
                                  stride=(1, 1), pad=(1, 1), with_bias=False,
                                  w_init=I.PytorchConv2dWeightInitializer(inmaps=encoded_image.shape[1], outmaps=1024, kernel=(3, 3)))
        semantic = F.relu(semantic)

    with nn.parameter_scope('fc_and_tile1'):
        semantic_shape = semantic.shape
        text_projection1 = _fc_and_tile(
            encoded_text, fc_outmaps=1024, out_shape=semantic_shape[-2:])
        if text_projection1.shape[0] == 1:
            semantic = semantic * text_projection1
        else:
            # We know that encoded image's original shape is (B, R, C, H, W) and is merged to (B * R, C, H, W)
            batch_size = text_projection1.shape[0]
            semantic = F.reshape(semantic, shape=(
                batch_size, semantic.shape[0] / batch_size, *semantic.shape[1:]))
            text_projection1 = F.reshape(text_projection1, shape=(
                batch_size, 1, *text_projection1.shape[1:]))
            semantic = semantic * text_projection1
            semantic = F.reshape(semantic, shape=(
                semantic.shape[0] * semantic.shape[1], *semantic.shape[2:]))
    with nn.parameter_scope('up_block1'):
        semantic = _up_block(
            semantic, image_features[2], outmaps=512, midmaps=1024, training=training)
    with nn.parameter_scope('conv_fusion1'):
        semantic = conv_fusion(semantic, spatial[0], outmaps=512)

    with nn.parameter_scope('fc_and_tile2'):
        semantic_shape = semantic.shape
        text_projection2 = _fc_and_tile(
            encoded_text, fc_outmaps=512, out_shape=semantic_shape[-2:])
        if text_projection2.shape[0] == 1:
            semantic = semantic * text_projection2
        else:
            # We know that encoded image's original shape is (B, R, C, H, W) and is merged to (B * R, C, H, W)
            batch_size = text_projection2.shape[0]
            semantic = F.reshape(semantic, shape=(
                batch_size, semantic.shape[0] / batch_size, *semantic.shape[1:]))
            text_projection2 = F.reshape(text_projection2, shape=(
                batch_size, 1, *text_projection2.shape[1:]))
            semantic = semantic * text_projection2
            semantic = F.reshape(semantic, shape=(
                semantic.shape[0] * semantic.shape[1], *semantic.shape[2:]))
    with nn.parameter_scope('up_block2'):
        semantic = _up_block(
            semantic, image_features[1], outmaps=256, midmaps=512, training=training)
    with nn.parameter_scope('conv_fusion2'):
        semantic = conv_fusion(semantic, spatial[1], outmaps=256)

    with nn.parameter_scope('fc_and_tile3'):
        semantic_shape = semantic.shape
        text_projection3 = _fc_and_tile(
            encoded_text, fc_outmaps=256, out_shape=semantic_shape[-2:])
        if text_projection3.shape[0] == 1:
            semantic = semantic * text_projection3
        else:
            # We know that encoded image's original shape is (B, R, C, H, W) and is merged to (B * R, C, H, W)
            batch_size = text_projection3.shape[0]
            semantic = F.reshape(semantic, shape=(
                batch_size, semantic.shape[0] / batch_size, *semantic.shape[1:]))
            text_projection3 = F.reshape(text_projection3, shape=(
                batch_size, 1, *text_projection3.shape[1:]))
            semantic = semantic * text_projection3
            semantic = F.reshape(semantic, shape=(
                semantic.shape[0] * semantic.shape[1], *semantic.shape[2:]))
    with nn.parameter_scope('up_block3'):
        semantic = _up_block(
            semantic, image_features[0], outmaps=128, midmaps=256, training=training)
    with nn.parameter_scope('conv_fusion3'):
        semantic = conv_fusion(semantic, spatial[2], outmaps=128)

    with nn.parameter_scope('conv_and_identity1'):
        semantic = F.relu(_conv_block(
            semantic, outmaps=[64, 64, 64], stride=(1, 1)))
        semantic = F.relu(_identity_block(
            semantic, outmaps=[64, 64, 64], stride=(1, 1)))
        semantic = _upsample(semantic, scale=2.0)
    with nn.parameter_scope('conv_fusion4'):
        semantic = conv_fusion(semantic, spatial[3], outmaps=64)

    with nn.parameter_scope('conv_and_identity2'):
        semantic = F.relu(_conv_block(
            semantic, outmaps=[32, 32, 32], stride=(1, 1)))
        semantic = F.relu(_identity_block(
            semantic, outmaps=[32, 32, 32], stride=(1, 1)))
        semantic = _upsample(semantic, scale=2.0)
    with nn.parameter_scope('conv_fusion5'):
        semantic = conv_fusion(semantic, spatial[4], outmaps=32)

    with nn.parameter_scope('conv_and_identity3'):
        semantic = F.relu(_conv_block(
            semantic, outmaps=[16, 16, 16], stride=(1, 1)))
        semantic = F.relu(_identity_block(
            semantic, outmaps=[16, 16, 16], stride=(1, 1)))
        semantic = _upsample(semantic, scale=2.0)
    with nn.parameter_scope('conv_fusion6'):
        semantic = conv_fusion(semantic, spatial[5], outmaps=16)

    with nn.parameter_scope('semantic_conv2'):
        semantic = PF.convolution(semantic, outmaps=outdim, kernel=(1, 1),
                                  w_init=I.PytorchConv2dWeightInitializer(
                                      inmaps=semantic.shape[1], outmaps=outdim, kernel=(1, 1)),
                                  b_init=I.PytorchConv2dBiasInitializer(inmaps=semantic.shape[1], outmaps=outdim, kernel=(1, 1)))
    semantic = _downsample(semantic, output_size=output_size)

    return semantic


def bottleneck(x, inplanes, planes, stride=1, expansion=4, training=False):
    identity = x

    with nn.parameter_scope('conv1'):
        out = PF.convolution(x, outmaps=planes, kernel=(1, 1), with_bias=False, stride=(1, 1),
                             w_init=I.PytorchConv2dWeightInitializer(inmaps=x.shape[1], outmaps=planes, kernel=(1, 1)))
    with nn.parameter_scope('bn1'):
        out = PF.batch_normalization(out, batch_stat=training)
        out = F.relu(out)
    with nn.parameter_scope('conv2'):
        out = PF.convolution(out, outmaps=planes, kernel=(3, 3), with_bias=False, stride=(1, 1), pad=(1, 1),
                             w_init=I.PytorchConv2dWeightInitializer(inmaps=out.shape[1], outmaps=planes, kernel=(3, 3)))
    with nn.parameter_scope('bn2'):
        out = PF.batch_normalization(out, batch_stat=training)
        out = F.relu(out)
    if stride > 1:
        out = F.average_pooling(out, kernel=(stride, stride))
    with nn.parameter_scope('conv3'):
        out = PF.convolution(out, outmaps=planes*expansion, kernel=(1, 1), with_bias=False, stride=(1, 1),
                             w_init=I.PytorchConv2dWeightInitializer(inmaps=out.shape[1], outmaps=planes*expansion, kernel=(1, 1)))
    with nn.parameter_scope('bn3'):
        out = PF.batch_normalization(out, batch_stat=training)

    if stride > 1 or inplanes != planes * expansion:
        # downsample
        # NOTE: stride size is used as kernel size in original code
        downsample = F.average_pooling(x, kernel=(stride, stride))

        with nn.parameter_scope('downsample'):
            with nn.parameter_scope('conv4'):
                downsample = PF.convolution(downsample, outmaps=planes*expansion,
                                            kernel=(1, 1), with_bias=False, stride=(1, 1),
                                            w_init=I.PytorchConv2dWeightInitializer(inmaps=downsample.shape[1], outmaps=planes*expansion, kernel=(1, 1)))
            with nn.parameter_scope('bn4'):
                downsample = PF.batch_normalization(
                    downsample, batch_stat=training)

        identity = downsample

    out += identity
    return F.relu(out)


def attention_pool_2d(x, spacial_dim, embed_dim, num_heads, output_dim=None):
    x = x.reshape(shape=(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
    x = F.transpose(x, axes=(2, 0, 1))  # NCHW -> (HW)NC
    x_mean = F.mean(x, axis=0, keepdims=True)
    x = F.concatenate(x_mean, x, axis=0)  # (HW+1)NC

    with nn.parameter_scope('positional_embedding'):
        embedding_shape = (spacial_dim ** 2 + 1, embed_dim)
        positional_embedding = nn.parameter.get_parameter_or_create(
            name='W', shape=embedding_shape)
        positional_embedding = positional_embedding.reshape(
            (x.shape[0], x.shape[1], embed_dim))
        x = x + positional_embedding

    k_proj_weight = nn.parameter.get_parameter_or_create(
        "k_proj/W", shape=(embed_dim, embed_dim))
    k_proj_bias = nn.parameter.get_parameter_or_create(
        "k_proj/b", shape=(embed_dim,))
    q_proj_weight = nn.parameter.get_parameter_or_create(
        "q_proj/W", shape=(embed_dim, embed_dim))
    q_proj_bias = nn.parameter.get_parameter_or_create(
        "q_proj/b", shape=(embed_dim,))
    v_proj_weight = nn.parameter.get_parameter_or_create(
        "v_proj/W", shape=(embed_dim, embed_dim))
    v_proj_bias = nn.parameter.get_parameter_or_create(
        "v_proj/b", shape=(embed_dim,))
    out_proj_weight = nn.parameter.get_parameter_or_create(
        "c_proj/W", shape=(output_dim or embed_dim, embed_dim))
    out_proj_bias = nn.parameter.get_parameter_or_create(
        "c_proj/b", shape=(output_dim or embed_dim,))

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


def positional_embedding(x, embedding_shape):
    positional_embedding = nn.parameter.get_parameter_or_create(
        name='W', shape=embedding_shape)
    positional_embedding = positional_embedding.reshape(
        (1, *positional_embedding.shape))
    return x + positional_embedding


def text_projection(x, width, embed_dim):
    text_projection = nn.parameter.get_parameter_or_create(
        name='W', shape=(width, embed_dim))
    text_projection = text_projection.reshape((1, width, embed_dim))
    return F.batch_matmul(x, text_projection)


def transformer(x, width, layers, heads, attn_mask=None):
    for i in range(layers):
        x = residual_attention_block(x, width, heads, i, attn_mask)
    return x


def residual_attention_block(x, d_model, num_heads, res_num, attn_mask=None):
    with nn.parameter_scope(f"resblocks/{res_num}"):
        with nn.parameter_scope('layer_norm1'):
            x_norm1 = PF.layer_normalization(x, batch_axis=(0, 1))
        with nn.parameter_scope('multi_head_attention'):
            x = x + multi_head_attention(x_norm1, x_norm1, x_norm1, d_model, num_heads,
                                         need_weights=False, attn_mask=attn_mask)
        with nn.parameter_scope('layer_norm2'):
            x_norm2 = PF.layer_normalization(x, batch_axis=(0, 1))
        with nn.parameter_scope('mlp'):
            x = x + mlp(x_norm2, d_model)
    return x


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
        with nn.parameter_scope('in_projection'):
            q, k, v = _in_projection_packed(
                query, key, value, outmaps=d_model*3)
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
    # attn_output: (L_T, B, E_v)
    attn_output = F.reshape(F.transpose(
        attn_output, (1, 0, 2)), (tgt_len, batch_size, embed_dim))

    if out_proj_weight is None and out_proj_bias is None:
        out_proj_weight = nn.parameter.get_parameter_or_create(
            name="attn/out_proj/W", shape=(d_model, d_model))
        out_proj_bias = nn.parameter.get_parameter_or_create(
            name="attn/out_proj/b", shape=(d_model,))
        out_proj_weight = F.transpose(out_proj_weight, (1, 0))
        attn_output = F.affine(
            attn_output, out_proj_weight, out_proj_bias, base_axis=2)
    elif out_proj_weight is None:
        raise NotImplementedError
    elif out_proj_bias is None:
        raise NotImplementedError
    else:
        out_proj_weight = F.transpose(out_proj_weight, (1, 0))
        attn_output = F.affine(
            attn_output, out_proj_weight, out_proj_bias, base_axis=2)
    return attn_output


def mlp(x, d_model):
    with nn.parameter_scope('affine1'):
        h = PF.affine(x, d_model * 4, base_axis=2)
    h = _quick_gelu(h)
    with nn.parameter_scope('affine2'):
        h = PF.affine(h, d_model, base_axis=2)
    return h


def _quick_gelu(x):
    return x * F.sigmoid(1.702 * x)


def _in_projection_packed(q, k, v, outmaps):
    if k is v:
        if q is k:
            # self-attention
            with nn.parameter_scope('affine'):
                to_ret = PF.affine(q, n_outmaps=outmaps, base_axis=2)
            ind = -(-to_ret.size_from_axis(2) // 3)
            a, b, c = to_ret.shape
            return (F.slice(to_ret, (0, 0, 0), (a, b, ind)),
                    F.slice(to_ret, (0, 0, ind), (a, b, ind*2)),
                    F.slice(to_ret, (0, 0, ind*2), (a, b, c)))
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


def _conv_block(x, outmaps, stride):
    with nn.parameter_scope('conv1'):
        h = PF.convolution(x, outmaps=outmaps[0], kernel=(1, 1), stride=(1, 1), with_bias=False,
                           w_init=I.PytorchConv2dWeightInitializer(inmaps=x.shape[1], outmaps=outmaps[0], kernel=(1, 1)))
        h = F.relu(h)
    with nn.parameter_scope('conv2'):
        h = PF.convolution(h, outmaps=outmaps[1], kernel=(3, 3),
                           stride=stride, pad=(1, 1), dilation=(1, 1), with_bias=False,
                           w_init=I.PytorchConv2dWeightInitializer(inmaps=h.shape[1], outmaps=outmaps[1], kernel=(3, 3)))
        h = F.relu(h)
    with nn.parameter_scope('conv3'):
        h = PF.convolution(h, outmaps=outmaps[2], kernel=(1, 1), stride=(1, 1), with_bias=False,
                           w_init=I.PytorchConv2dWeightInitializer(inmaps=h.shape[1], outmaps=outmaps[2], kernel=(1, 1)))
    with nn.parameter_scope('shortcut'):
        # not mentioned in the paper but exists in the author's code
        # https://github.com/cliport/cliport
        h = h + PF.convolution(x, outmaps=outmaps[2], kernel=(1, 1), stride=stride, with_bias=False,
                               w_init=I.PytorchConv2dWeightInitializer(inmaps=h.shape[1], outmaps=outmaps[2], kernel=(1, 1)))
    return h


def _identity_block(x, outmaps, stride):
    with nn.parameter_scope('identity1'):
        h = PF.convolution(x, outmaps=outmaps[0], kernel=(1, 1), stride=(1, 1), with_bias=False,
                           w_init=I.PytorchConv2dWeightInitializer(inmaps=x.shape[1], outmaps=outmaps[0], kernel=(1, 1)))
        h = F.relu(h)
    with nn.parameter_scope('identity2'):
        h = PF.convolution(h, outmaps=outmaps[1], kernel=(3, 3),
                           stride=stride, pad=(1, 1), dilation=(1, 1), with_bias=False,
                           w_init=I.PytorchConv2dWeightInitializer(inmaps=h.shape[1], outmaps=outmaps[1], kernel=(3, 3)))
        h = F.relu(h)
    with nn.parameter_scope('identity3'):
        h = PF.convolution(h, outmaps=outmaps[2], kernel=(1, 1), stride=(1, 1), with_bias=False,
                           w_init=I.PytorchConv2dWeightInitializer(inmaps=h.shape[1], outmaps=outmaps[2], kernel=(1, 1)))
    return h + x  # add x for residual connection


def _fc_and_tile(x, fc_outmaps, out_shape):
    with nn.parameter_scope('linear1'):
        x = PF.affine(x, n_outmaps=fc_outmaps,
                      w_init=I.PytorchLinearWeightInitializer(
                          inmaps=x.shape[-1], outmaps=fc_outmaps),
                      b_init=I.PytorchLinearBiasInitializer(inmaps=x.shape[-1], outmaps=fc_outmaps))
    x = F.reshape(x, shape=(*x.shape, 1, 1))
    x = F.tile(x, reps=(1, 1, *out_shape))
    return x


def _up_block(x, y, outmaps, midmaps=None, training=False):
    x = _upsample(x, scale=2.0)

    diffH = y.shape[2] - x.shape[2]
    diffW = y.shape[3] - x.shape[3]

    pad_width = (diffH // 2, diffH - diffH // 2,
                 diffW // 2, diffW - diffW // 2)
    x = F.pad(x, pad_width)
    # Concat along channel
    h = F.concatenate(y, x, axis=1)
    if not midmaps:
        midmaps = outmaps
    with nn.parameter_scope('conv1'):
        h = PF.convolution(h, outmaps=midmaps, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                           w_init=I.PytorchConv2dWeightInitializer(
                               inmaps=h.shape[1], outmaps=midmaps, kernel=(3, 3)),
                           b_init=I.PytorchConv2dBiasInitializer(inmaps=h.shape[1], outmaps=midmaps, kernel=(3, 3)))
    with nn.parameter_scope('bn1'):
        h = PF.batch_normalization(h, batch_stat=training)
        h = F.relu(h)
    with nn.parameter_scope('conv2'):
        h = PF.convolution(h, outmaps=outmaps, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                           w_init=I.PytorchConv2dWeightInitializer(
                               inmaps=h.shape[1], outmaps=outmaps, kernel=(3, 3)),
                           b_init=I.PytorchConv2dBiasInitializer(inmaps=h.shape[1], outmaps=outmaps, kernel=(3, 3)))
    with nn.parameter_scope('bn2'):
        h = PF.batch_normalization(h, batch_stat=training)
        h = F.relu(h)
    return h


def _upsample(x, scale=None, output_size=None):
    assert scale > 1.0
    if isinstance(scale, float):
        scale = (scale, scale)
    h = F.interpolate(x, scale=scale, half_pixel=True)
    return h


def _downsample(x, scale=None, output_size=None):
    h = F.interpolate(x, scale=scale, output_size=output_size, half_pixel=True)
    return h


def conv_fusion(x1, x2, outmaps):
    x = F.concatenate(x1, x2, axis=1)  # concat along channel axis
    x = F.relu(x)
    with nn.parameter_scope('conv1'):
        x = PF.convolution(x, outmaps=outmaps, kernel=(1, 1), with_bias=False,
                           w_init=I.PytorchConv2dWeightInitializer(inmaps=x.shape[1], outmaps=outmaps, kernel=(1, 1)))
    return x


def add_fusion(x1, x2):
    return x1 + x2
