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

from .common import *
from .progress_tracker import MonitorManager

from network.mlp import MLP
from data_iterator import get_photo_tourism_dataiterator
from data_iterator.get_data import get_data

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.initializer as I

from nnabla.utils.image_utils import imsave

import os
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


def get_radiance_field(sample_points, view_directions, app_emb, trans_emb, encode_position_function, encode_direction_function,
                       chunksize, scope_name='nerf', use_transient=True):
    flatten_sample_points = F.reshape(sample_points, (-1, 3))
    input_encodings = encode_position_function(flatten_sample_points)
    input_dim = input_encodings.shape[-1]
    input_views_dim = 0

    if view_directions is not None:
        view_directions = F.reshape(
            view_directions, view_directions.shape[:view_directions.ndim-1]+(1, 3))
        view_directions = F.broadcast(view_directions, sample_points.shape)
        flatten_view_directions = F.reshape(
            view_directions, (-1, 3))
        direction_encodings = encode_direction_function(
            flatten_view_directions)
        input_encodings = F.concatenate(
            input_encodings, direction_encodings, axis=1)
        input_views_dim = direction_encodings.shape[-1]

    flatten_radiance_field = []
    num_batches = (input_encodings.shape[0]//chunksize) + 1

    for i in range(num_batches):
        if i != num_batches-1:
            flatten_radiance_field.append(MLP(input_encodings[i*chunksize: (
                i+1)*chunksize], app_emb=app_emb, trans_emb=trans_emb, input_pos_dim=input_dim, input_views_dim=input_views_dim, scope_name=scope_name, use_transient=use_transient))
        else:
            if input_encodings.shape[0] - (num_batches-1)*chunksize == 0:
                continue

            # Need to handle this case for trans_emb and app_emb, in case different indices form the embeddings
            chunksize_residual = input_encodings.shape[0] - (
                num_batches-1)*chunksize
            if app_emb is not None:
                app_emb = app_emb[:chunksize_residual, :]
            if trans_emb is not None:
                trans_emb = trans_emb[:chunksize_residual, :]

            flatten_radiance_field.append(MLP(input_encodings[(
                num_batches-1)*chunksize:], app_emb=app_emb, trans_emb=trans_emb, input_pos_dim=input_dim, input_views_dim=input_views_dim, scope_name=scope_name, use_transient=use_transient))

    flatten_radiance_field = F.concatenate(*flatten_radiance_field, axis=0) if len(
        flatten_radiance_field) > 1 else flatten_radiance_field[0]
    radiance_field = F.reshape(
        flatten_radiance_field, sample_points.shape[:-1]+(-1,))

    return radiance_field


def forward_pass(ray_directions, ray_origins, near_plane, far_plane, app_emb, trans_emb, encode_position_function, encode_direction_function,
                 config, use_transient, hwf=None, image=None):

    if encode_direction_function is not None:
        view_directions = ray_directions
        view_directions = view_directions / \
            F.norm(view_directions, p=2, axis=-1, keepdims=True)
    else:
        view_directions = None

    # For Forward facing dataset like LLFF, scene discovery is done in NDC system
    if config.train.use_ndc:
        ray_origins, ray_directions = ndc_rays(
            hwf[0], hwf[1], hwf[2], 1, ray_origins, ray_directions)

    if isinstance(ray_directions, nn.Variable):
        randomize = True
    else:  # Inference
        randomize = False

    sample_points, depth_values = compute_sample_points_from_rays(
        ray_origins, ray_directions, near_plane, far_plane, config.train.num_samples_course, randomize=randomize)

    radiance_field = get_radiance_field(
        sample_points, view_directions, app_emb, trans_emb, encode_position_function, encode_direction_function, config.train.chunksize_course, 'nerf_coarse', use_transient=use_transient)

    if use_transient:
        rgb_map_course, weights_course = volume_rendering_transient(radiance_field, ray_origins, depth_values,
                                                                    return_weights=True, white_bkgd=config.train.white_bkgd, raw_noise_std=config.train.raw_noise_std)

    else:
        (rgb_map_course, depth_map_course, acc_map_course, disp_map_course, weights_course) = \
            volumetric_rendering(radiance_field, ray_origins, depth_values,
                                 return_weights=True, white_bkgd=config.train.white_bkgd, raw_noise_std=config.train.raw_noise_std)

    # Get fine depth values
    num_additional_points = config.train.num_samples_fine - \
        config.train.num_samples_course
    if randomize is False:
        depth_values = F.broadcast(
            depth_values, (ray_origins.shape[0], depth_values.shape[-1]))

    depth_values_mid = 0.5*(depth_values[..., 1:] + depth_values[..., :-1])
    depth_samples = sample_pdf(
        depth_values_mid, weights_course[..., 1:-1], num_additional_points, det=not randomize)

    if isinstance(depth_samples, nn.Variable):
        depth_samples = depth_samples.get_unlinked_variable(need_grad=False)
    elif isinstance(depth_samples, nn.NdArray):
        pass
    elif isinstance(depth_samples, np.ndarray):
        if isinstance(radiance_field, nn.Variable):
            depth_samples = nn.Variable.from_numpy_array(depth_samples)
        else:
            depth_samples = nn.NdArray.from_numpy_array(depth_samples)
    else:
        raise NotImplementedError

    depth_values = F.sort(F.concatenate(
        depth_values, depth_samples, axis=depth_samples.ndim-1), axis=depth_values.ndim-1)

    sample_points = ray_origins[..., None, :] + \
        ray_directions[..., None, :]*depth_values[..., :, None]
    radiance_field = get_radiance_field(
        sample_points, view_directions, app_emb, trans_emb, encode_position_function, encode_direction_function, config.train.chunksize_fine, 'nerf_fine', use_transient=use_transient)

    if use_transient:
        rgb_map_fine, weights_fine, static_rgb_map_fine, transient_rgb_map_fine, beta = \
            volume_rendering_transient(radiance_field, ray_origins, depth_values,
                                       return_weights=False, white_bkgd=config.train.white_bkgd, raw_noise_std=config.train.raw_noise_std)
    else:
        rgb_map_fine, depth_map_fine, acc_map_fine, disp_map_fine, weights_fine = \
            volumetric_rendering(radiance_field, ray_origins, depth_values,
                                 return_weights=True, white_bkgd=config.train.white_bkgd, raw_noise_std=config.train.raw_noise_std)

    if use_transient:
        static_sigma = radiance_field[..., 3]
        transient_sigma = radiance_field[..., 7]

        return rgb_map_course, rgb_map_fine, static_rgb_map_fine, transient_rgb_map_fine, beta, static_sigma, transient_sigma

    else:
        return rgb_map_course, depth_map_course, disp_map_course, acc_map_course, rgb_map_fine, depth_map_fine, disp_map_fine, acc_map_fine


def train_nerf(config, comm, model, dataset='blender'):

    use_transient = False
    use_embedding = False

    if model == 'wild':
        use_transient = True
        use_embedding = True
    elif model == 'uncertainty':
        use_transient = True
    elif model == 'appearance':
        use_embedding = True

    save_results_dir = config.log.save_results_dir
    os.makedirs(save_results_dir, exist_ok=True)

    train_loss_dict = {
        'train_coarse_loss': 0.0,
        'train_fine_loss': 0.0,
        'train_total_loss': 0.0,
    }

    test_metric_dict = {
        'test_loss': 0.0,
        'test_psnr': 0.0
    }

    monitor_manager = MonitorManager(
        train_loss_dict, test_metric_dict, save_results_dir)

    if dataset != 'phototourism':
        images, poses, _, hwf, i_test, i_train, near_plane, far_plane = get_data(
            config)
        height, width, focal_length = hwf
    else:
        di = get_photo_tourism_dataiterator(config, 'train', comm)
        val_di = get_photo_tourism_dataiterator(config, 'val', comm)

    if model != 'vanilla':
        if dataset != 'phototourism':
            config.train.n_vocab = max(np.max(i_train), np.max(i_test)) + 1
        print(
            f'Setting Vocabulary size of embedding as {config.train.n_vocab}')

    if dataset != 'phototourism':
        if model in ['vanilla']:
            if comm is not None:
                # uncomment the following line to test on fewer images
                i_test = i_test[3*comm.rank:3*(comm.rank+1)]
                pass
            else:
                # uncomment the following line to test on fewer images
                i_test = i_test[:3]
                pass
        else:
            # i_test = i_train[0:5]
            i_test = [i*(comm.rank+1) for i in range(5)]
    else:
        i_test = [1]

    encode_position_function = get_encoding_function(
        config.train.num_encodings_position, True, True)
    if config.train.use_view_directions:
        encode_direction_function = get_encoding_function(
            config.train.num_encodings_direction, True, True)
    else:
        encode_direction_function = None

    lr = config.solver.lr
    num_decay_steps = config.solver.lr_decay_step * 1000
    lr_decay_factor = config.solver.lr_decay_factor
    solver = S.Adam(alpha=lr)

    load_solver_state = False
    if config.checkpoint.param_path is not None:
        nn.load_parameters(config.checkpoint.param_path)
        load_solver_state = True

    if comm is not None:
        num_decay_steps /= comm.n_procs
        comm_size = comm.n_procs
    else:
        comm_size = 1
    pbar = trange(config.train.num_iterations//comm_size,
                  disable=(comm is not None and comm.rank > 0))

    for i in pbar:

        if dataset != 'phototourism':

            idx = np.random.choice(i_train)
            image = nn.Variable.from_numpy_array(images[idx][None, :, :, :3])
            pose = nn.Variable.from_numpy_array(poses[idx])

            ray_directions, ray_origins = get_ray_bundle(
                height, width, focal_length, pose)

            grid = get_direction_grid(
                width, height, focal_length, return_ij_2d_grid=True)
            grid = F.reshape(grid, (-1, 2))

            select_inds = np.random.choice(
                grid.shape[0], size=[config.train.num_rand_points], replace=False)
            select_inds = F.gather_nd(grid, select_inds[None, :])
            select_inds = F.transpose(select_inds, (1, 0))

            embed_inp = nn.Variable.from_numpy_array(
                np.full((config.train.chunksize_fine,), idx, dtype=int))

            ray_origins = F.gather_nd(ray_origins, select_inds)
            ray_directions = F.gather_nd(ray_directions, select_inds)

            image = F.gather_nd(image[0], select_inds)

        else:
            rays, embed_inp, image = di.next()
            ray_origins = nn.Variable.from_numpy_array(rays[:, :3])
            ray_directions = nn.Variable.from_numpy_array(rays[:, 3:6])
            near_plane = nn.Variable.from_numpy_array(rays[:, 6])
            far_plane = nn.Variable.from_numpy_array(rays[:, 7])

            embed_inp = nn.Variable.from_numpy_array(embed_inp)
            image = nn.Variable.from_numpy_array(image)

            hwf = None

        app_emb, trans_emb = None, None
        if use_embedding:
            with nn.parameter_scope('embedding_a'):
                app_emb = PF.embed(
                    embed_inp, config.train.n_vocab, config.train.n_app)

        if use_transient:
            with nn.parameter_scope('embedding_t'):
                trans_emb = PF.embed(
                    embed_inp, config.train.n_vocab, config.train.n_trans)

        if use_transient:
            rgb_map_course, rgb_map_fine, static_rgb_map_fine, transient_rgb_map_fine, beta, static_sigma, transient_sigma = forward_pass(ray_directions, ray_origins, near_plane, far_plane,
                                                                                                                                          app_emb, trans_emb, encode_position_function, encode_direction_function, config, use_transient, hwf=hwf, image=image)
            course_loss = 0.5 * F.mean(F.squared_error(rgb_map_course, image))
            fine_loss = 0.5 * F.mean(F.squared_error(rgb_map_fine, image) / F.reshape(
                F.pow_scalar(beta, 2), beta.shape+(1,)))
            beta_reg_loss = 3 + F.mean(F.log(beta))
            sigma_trans_reg_loss = 0.01 * F.mean(transient_sigma)
            loss = course_loss + fine_loss + beta_reg_loss + sigma_trans_reg_loss
        else:
            rgb_map_course, _, _, _, rgb_map_fine, _, _, _ = forward_pass(ray_directions, ray_origins, near_plane, far_plane,
                                                                          app_emb, trans_emb, encode_position_function, encode_direction_function, config, use_transient, hwf=hwf)
            course_loss = F.mean(F.squared_error(rgb_map_course, image))
            fine_loss = F.mean(F.squared_error(rgb_map_fine, image))
            loss = course_loss + fine_loss

        pbar.set_description(
            f'Total: {np.around(loss.d, 4)}, Course: {np.around(course_loss.d, 4)}, Fine: {np.around(fine_loss.d, 4)}')

        solver.set_parameters(nn.get_parameters(),
                              reset=False, retain_state=True)
        if load_solver_state:
            solver.load_states(config['checkpoint']['solver_path'])
            load_solver_state = False

        solver.zero_grad()

        loss.backward(clear_buffer=True)

        # Exponential LR decay
        if dataset != 'phototourism':
            lr_factor = (lr_decay_factor ** ((i) / num_decay_steps))
            solver.set_learning_rate(lr * lr_factor)
        else:
            if i % num_decay_steps == 0 and i != 0:
                solver.set_learning_rate(lr * lr_decay_factor)

        if comm is not None:
            params = [x.grad for x in nn.get_parameters().values()]
            comm.all_reduce(params, division=False, inplace=True)
        solver.update()

        if ((i % config.train.save_interval == 0 or i == config.train.num_iterations-1) and i != 0) and (comm is not None and comm.rank == 0):
            nn.save_parameters(os.path.join(save_results_dir, f'iter_{i}.h5'))
            solver.save_states(os.path.join(
                save_results_dir, f'solver_iter_{i}.h5'))

        if (i % config.train.test_interval == 0 or i == config.train.num_iterations-1) and i != 0:
            avg_psnr, avg_mse = 0.0, 0.0
            for i_t in trange(len(i_test)):

                if dataset != 'phototourism':
                    idx_test = i_test[i_t]
                    image = nn.NdArray.from_numpy_array(
                        images[idx_test][None, :, :, :3])
                    pose = nn.NdArray.from_numpy_array(poses[idx_test])

                    ray_directions, ray_origins = get_ray_bundle(
                        height, width, focal_length, pose)

                    ray_directions = F.reshape(
                        ray_directions, (-1, 3))
                    ray_origins = F.reshape(
                        ray_origins, (-1, 3))

                    embed_inp = nn.NdArray.from_numpy_array(
                        np.full((config.train.chunksize_fine,), idx_test, dtype=int))

                else:
                    rays, embed_inp, image = val_di.next()
                    ray_origins = nn.NdArray.from_numpy_array(rays[0, :, :3])
                    ray_directions = nn.NdArray.from_numpy_array(
                        rays[0, :, 3:6])
                    near_plane_ = nn.NdArray.from_numpy_array(rays[0, :, 6])
                    far_plane_ = nn.NdArray.from_numpy_array(rays[0, :, 7])

                    embed_inp = nn.NdArray.from_numpy_array(
                        embed_inp[0, :config.train.chunksize_fine])
                    image = nn.NdArray.from_numpy_array(
                        image[0].transpose(1, 2, 0))
                    image = F.reshape(image, (1,)+image.shape)
                    idx_test = 1

                app_emb, trans_emb = None, None
                if use_embedding:
                    with nn.parameter_scope('embedding_a'):
                        app_emb = PF.embed(
                            embed_inp, config.train.n_vocab, config.train.n_app)

                if use_transient:
                    with nn.parameter_scope('embedding_t'):
                        trans_emb = PF.embed(
                            embed_inp, config.train.n_vocab, config.train.n_trans)

                num_ray_batches = ray_directions.shape[0]//config.train.ray_batch_size+1

                if use_transient:
                    rgb_map_fine_list, static_rgb_map_fine_list, transient_rgb_map_fine_list = [], [], []
                else:
                    rgb_map_fine_list, depth_map_fine_list = [], []

                for r_idx in trange(num_ray_batches):
                    if r_idx != num_ray_batches-1:
                        ray_d, ray_o = ray_directions[r_idx*config.train.ray_batch_size:(
                            r_idx+1)*config.train.ray_batch_size], ray_origins[r_idx*config.train.ray_batch_size:(r_idx+1)*config.train.ray_batch_size]

                        if dataset == 'phototourism':
                            near_plane = near_plane_[r_idx*config.train.ray_batch_size:(
                                r_idx+1)*config.train.ray_batch_size]
                            far_plane = far_plane_[r_idx*config.train.ray_batch_size:(
                                r_idx+1)*config.train.ray_batch_size]

                    else:
                        if ray_directions.shape[0] - (num_ray_batches-1)*config.train.ray_batch_size == 0:
                            break
                        ray_d, ray_o = ray_directions[r_idx*config.train.ray_batch_size:,
                                                      :], ray_origins[r_idx*config.train.ray_batch_size:, :]
                        if dataset == 'phototourism':
                            near_plane = near_plane_[
                                r_idx*config.train.ray_batch_size:]
                            far_plane = far_plane_[
                                r_idx*config.train.ray_batch_size:]

                    if use_transient:
                        rgb_map_course, rgb_map_fine, static_rgb_map_fine, transient_rgb_map_fine, beta, static_sigma, transient_sigma = forward_pass(ray_d, ray_o, near_plane, far_plane,
                                                                                                                                                      app_emb, trans_emb, encode_position_function, encode_direction_function, config, use_transient, hwf=hwf)

                        rgb_map_fine_list.append(rgb_map_fine)
                        static_rgb_map_fine_list.append(static_rgb_map_fine)
                        transient_rgb_map_fine_list.append(
                            transient_rgb_map_fine)

                    else:
                        _, _, _, _, rgb_map_fine, depth_map_fine, _, _ = \
                            forward_pass(ray_d, ray_o, near_plane, far_plane, app_emb, trans_emb,
                                         encode_position_function, encode_direction_function, config, use_transient, hwf=hwf)

                        rgb_map_fine_list.append(rgb_map_fine)
                        depth_map_fine_list.append(depth_map_fine)

                if use_transient:
                    rgb_map_fine = F.concatenate(*rgb_map_fine_list, axis=0)
                    static_rgb_map_fine = F.concatenate(
                        *static_rgb_map_fine_list, axis=0)
                    transient_rgb_map_fine = F.concatenate(
                        *transient_rgb_map_fine_list, axis=0)

                    rgb_map_fine = F.reshape(rgb_map_fine, image[0].shape)
                    static_rgb_map_fine = F.reshape(
                        static_rgb_map_fine, image[0].shape)
                    transient_rgb_map_fine = F.reshape(
                        transient_rgb_map_fine, image[0].shape)
                    static_trans_img_to_save = np.concatenate((static_rgb_map_fine.data, np.ones(
                        (image[0].shape[0], 5, 3)), transient_rgb_map_fine.data), axis=1)
                    img_to_save = np.concatenate((image[0].data, np.ones(
                        (image[0].shape[0], 5, 3)), rgb_map_fine.data), axis=1)
                else:

                    rgb_map_fine = F.concatenate(*rgb_map_fine_list, axis=0)
                    depth_map_fine = F.concatenate(
                        *depth_map_fine_list, axis=0)

                    rgb_map_fine = F.reshape(rgb_map_fine, image[0].shape)
                    depth_map_fine = F.reshape(
                        depth_map_fine, image[0].shape[:-1])
                    img_to_save = np.concatenate((image[0].data, np.ones(
                        (image[0].shape[0], 5, 3)), rgb_map_fine.data), axis=1)

                filename = os.path.join(
                    save_results_dir, f'{i}_{idx_test}.png')
                try:
                    imsave(filename, np.clip(img_to_save, 0, 1),
                           channel_first=False)
                    print(f'Saved generation at {filename}')
                    if use_transient:
                        filename_static_trans = os.path.join(
                            save_results_dir, f's_t_{i}_{idx_test}.png')
                        imsave(filename_static_trans, np.clip(
                            static_trans_img_to_save, 0, 1), channel_first=False)

                    else:
                        filename_dm = os.path.join(
                            save_results_dir, f'dm_{i}_{idx_test}.png')
                        depth_map_fine = (depth_map_fine.data - depth_map_fine.data.min())/(
                            depth_map_fine.data.max() - depth_map_fine.data.min())
                        imsave(filename_dm,
                               depth_map_fine[:, :, None], channel_first=False)
                        plt.imshow(depth_map_fine.data)
                        plt.savefig(filename_dm)
                        plt.close()
                except:
                    pass

                avg_mse += F.mean(F.squared_error(rgb_map_fine, image[0])).data
                avg_psnr += (-10. *
                             np.log10(F.mean(F.squared_error(rgb_map_fine, image[0])).data))

            test_metric_dict['test_loss'] = avg_mse/len(i_test)
            test_metric_dict['test_psnr'] = avg_psnr/len(i_test)
            monitor_manager.add(i, test_metric_dict)
            print(
                f'Saved generations after {i} training iterations! Average PSNR: {avg_psnr/len(i_test)}. Average MSE: {avg_mse/len(i_test)}')
