# Copyright 2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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


def MLP(x, app_emb=None, trans_emb=None, num_layers_base=8, num_layers_static=1, num_layers_trans=4, hidden_dim=256, skips=[4], input_pos_dim=63, input_views_dim=27,
        encode_app=False, scope_name='nerf', use_transient=False):

    encode_app = False if scope_name == 'nerf_coarse' or app_emb is None else True
    use_transient = False if scope_name == 'nerf_coarse' else use_transient

    xyz, views = x[:, :input_pos_dim], x[:,
                                         input_pos_dim: input_pos_dim+input_views_dim]

    out = xyz
    for i in range(num_layers_base):
        if i in skips:
            out = F.concatenate(xyz, out, axis=1)
        with nn.parameter_scope(f'{scope_name}/xyz_encoding_{i+1}/0'):
            out = PF.affine(out, hidden_dim)
            out = F.relu(out)

    with nn.parameter_scope(f'{scope_name}/static_sigma/0'):
        static_sigma = PF.affine(out, 1)
        static_sigma = F.softplus(static_sigma)

    # transform base mlp output that will be concatenated with appearance embedding and view directions
    with nn.parameter_scope(f'{scope_name}/xyz_encoding_final'):
        xyz_encoding_final = PF.affine(out, hidden_dim)

    # try:
    xyz_views = F.concatenate(xyz_encoding_final, views, axis=1)

    if encode_app:
        xyz_views = F.concatenate(xyz_views, app_emb, axis=1)

    # transform xyz_views that will then be input to static rgb prediction model
    with nn.parameter_scope(f'{scope_name}/dir_encoding/0'):
        xyz_views = PF.affine(xyz_views, hidden_dim//2)
        xyz_views = F.relu(xyz_views)

    static_out = xyz_views
    for i in range(num_layers_static):
        with nn.parameter_scope(f'{scope_name}/static_rgb/{i}'):
            if i != num_layers_static-1:
                static_out = PF.affine(static_out, hidden_dim//2)
                static_out = F.relu(static_out)
            else:
                static_out = PF.affine(static_out, 3)
                static_rgb = F.sigmoid(static_out)

    output = F.concatenate(static_rgb, static_sigma, axis=1)

    if use_transient:
        trans_base_out = F.concatenate(xyz_encoding_final, trans_emb, axis=1)
        for i in range(num_layers_trans):
            with nn.parameter_scope(f'{scope_name}/transient_encoding/{i*2}'):
                trans_base_out = PF.affine(trans_base_out, hidden_dim//2)
                trans_base_out = F.relu(trans_base_out)

        with nn.parameter_scope(f'{scope_name}/transient_sigma/0'):
            trans_sigma = PF.affine(trans_base_out, 1)
            trans_sigma = F.softplus(trans_sigma)

        with nn.parameter_scope(f'{scope_name}/transient_rgb/0'):
            trans_rgb = PF.affine(trans_base_out, 3)
            trans_rgb = F.sigmoid(trans_rgb)

        with nn.parameter_scope(f'{scope_name}/transient_beta/0'):
            trans_beta = PF.affine(trans_base_out, 1)
            trans_beta = F.softplus(trans_beta)

        output = F.concatenate(
            output, trans_rgb, trans_sigma, trans_beta, axis=1)

    return output
