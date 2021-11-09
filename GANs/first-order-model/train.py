# Copyright 2021 Sony Corporation.
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

import os
import shutil
import argparse
import numpy as np
from typing import NamedTuple

import nnabla as nn
import nnabla.communicators as C
import nnabla.solvers as S
import nnabla.logger as logger

from nnabla.ext_utils import get_extension_context

from frames_dataset import frame_data_iterator
from keypoint_detector import detect_keypoint
from generator import occlusion_aware_generator
from discriminator import multiscale_discriminator
from model import get_image_pyramid, Transform, unlink_all, persistent_all
from utils import get_monitors, combine_images, read_yaml
from loss import perceptual_loss, lsgan_loss, feature_matching_loss, equivariance_value_loss, equivariance_jacobian_loss


class LossFlags(NamedTuple):
    use_perceptual_loss: bool
    use_gan_loss: bool
    use_feature_matching_loss: bool
    use_equivariance_value_loss: bool
    use_equivariance_jacobian_loss: bool


def get_loss_flags(train_params):
    if sum(train_params['loss_weights']['perceptual']) != 0:
        use_perceptual_loss = True
    else:
        use_perceptual_loss = False

    if train_params.loss_weights.generator_gan != 0:
        use_gan_loss = True
        if sum(train_params.loss_weights.feature_matching) != 0:
            use_feature_matching_loss = True
        else:
            use_feature_matching_loss = False
    else:
        use_gan_loss = False
        use_feature_matching_loss = False

    if train_params.loss_weights.equivariance_value != 0:
        use_equivariance_value_loss = True
    else:
        use_equivariance_value_loss = False

    if train_params.loss_weights.equivariance_jacobian != 0:
        use_equivariance_jacobian_loss = True
    else:
        use_equivariance_jacobian_loss = False

    loss_flags = LossFlags(use_perceptual_loss,
                           use_gan_loss,
                           use_feature_matching_loss,
                           use_equivariance_value_loss,
                           use_equivariance_jacobian_loss)
    return loss_flags


def setup_solvers(train_params):
    solver_dict = dict()

    solver_generator = S.Adam(alpha=train_params['lr_generator'],
                              beta1=0.5, beta2=0.999)
    with nn.parameter_scope("generator"):
        solver_generator.set_parameters(nn.get_parameters())
    solver_dict["generator"] = solver_generator

    solver_discriminator = S.Adam(train_params['lr_discriminator'],
                                  beta1=0.5, beta2=0.999)
    with nn.parameter_scope("discriminator"):
        solver_discriminator.set_parameters(nn.get_parameters())
    solver_dict["discriminator"] = solver_discriminator

    solver_kp_detector = S.Adam(train_params['lr_kp_detector'],
                                beta1=0.5, beta2=0.999)
    with nn.parameter_scope("kp_detector"):
        solver_kp_detector.set_parameters(nn.get_parameters())

    solver_dict["kp_detector"] = solver_kp_detector
    return solver_dict


def learning_rate_decay(solvers, gamma=0.1):
    for solver in solvers.values():
        lr = solver.learning_rate()
        solver.set_learning_rate(lr*gamma)


def save_parameters(current_epoch, log_dir, solvers):
    training_info_yaml = os.path.join(log_dir, "training_info.yaml")

    # save weights
    saved_parameter = os.path.join(
        log_dir, f"params_at_epoch_{current_epoch}.h5")
    nn.save_parameters(saved_parameter)

    # save solver's state
    for name, solver in solvers.items():
        saved_states = os.path.join(
            log_dir, f"state_{name}_at_epoch_{current_epoch}.h5")
        solver.save_states(saved_states)

    with open(training_info_yaml, "r", encoding="utf-8") as f:
        lines = f.readlines()[:-1]
    lines.append(f"saved_parameters: params_at_epoch_{current_epoch}.h5")

    # update the training info .yaml
    with open(training_info_yaml, "w", encoding="utf-8") as f:
        f.writelines(lines)


def train(args):

    # get context

    ctx = get_extension_context(args.context)
    comm = C.MultiProcessDataParalellCommunicator(ctx)
    comm.init()
    n_devices = comm.size
    mpi_rank = comm.rank
    device_id = mpi_rank
    ctx.device_id = str(device_id)
    nn.set_default_context(ctx)

    config = read_yaml(args.config)

    if args.info:
        config.monitor_params.info = args.info

    if comm.size == 1:
        comm = None
    else:
        # disable outputs from logger except its rank = 0
        if comm.rank > 0:
            import logging
            logger.setLevel(logging.ERROR)

    test = False
    train_params = config.train_params
    dataset_params = config.dataset_params
    model_params = config.model_params

    loss_flags = get_loss_flags(train_params)

    start_epoch = 0

    rng = np.random.RandomState(device_id)
    data_iterator = frame_data_iterator(root_dir=dataset_params.root_dir,
                                        frame_shape=dataset_params.frame_shape,
                                        id_sampling=dataset_params.id_sampling,
                                        is_train=True,
                                        random_seed=rng,
                                        augmentation_params=dataset_params.augmentation_params,
                                        batch_size=train_params['batch_size'],
                                        shuffle=True,
                                        with_memory_cache=False, with_file_cache=False)

    if n_devices > 1:
        data_iterator = data_iterator.slice(rng=rng,
                                            num_of_slices=comm.size,
                                            slice_pos=comm.rank)
        # workaround not to use memory cache
        data_iterator._data_source._on_memory = False
        logger.info("Disabled on memory data cache.")

    bs, h, w, c = [train_params.batch_size] + dataset_params.frame_shape
    source = nn.Variable((bs, c, h, w))
    driving = nn.Variable((bs, c, h, w))

    with nn.parameter_scope("kp_detector"):
        # kp_X = {"value": Variable((bs, 10, 2)), "jacobian": Variable((bs, 10, 2, 2))}

        kp_source = detect_keypoint(source,
                                    **model_params.kp_detector_params,
                                    **model_params.common_params,
                                    test=test, comm=comm)
        persistent_all(kp_source)

        kp_driving = detect_keypoint(driving,
                                     **model_params.kp_detector_params,
                                     **model_params.common_params,
                                     test=test, comm=comm)
        persistent_all(kp_driving)

    with nn.parameter_scope("generator"):
        generated = occlusion_aware_generator(source,
                                              kp_source=kp_source,
                                              kp_driving=kp_driving,
                                              **model_params.generator_params,
                                              **model_params.common_params,
                                              test=test, comm=comm)
        # generated is a dictionary containing;
        # 'mask': Variable((bs, num_kp+1, h/4, w/4)) when scale_factor=0.25
        # 'sparse_deformed': Variable((bs, num_kp + 1, num_channel, h/4, w/4))
        # 'occlusion_map': Variable((bs, 1, h/4, w/4))
        # 'deformed': Variable((bs, c, h, w))
        # 'prediction': Variable((bs, c, h, w)) Only this is fed to discriminator.

    generated["prediction"].persistent = True

    pyramide_real = get_image_pyramid(driving, train_params.scales,
                                      generated["prediction"].shape[1])
    persistent_all(pyramide_real)

    pyramide_fake = get_image_pyramid(generated['prediction'],
                                      train_params.scales,
                                      generated["prediction"].shape[1])
    persistent_all(pyramide_fake)

    total_loss_G = None  # dammy. defined temporarily
    loss_var_dict = {}

    # perceptual loss using VGG19 (always applied)
    if loss_flags.use_perceptual_loss:
        logger.info("Use Perceptual Loss.")
        scales = train_params.scales
        weights = train_params.loss_weights.perceptual
        vgg_param_path = train_params.vgg_param_path
        percep_loss = perceptual_loss(pyramide_real, pyramide_fake,
                                      scales, weights, vgg_param_path)
        percep_loss.persistent = True
        loss_var_dict['perceptual_loss'] = percep_loss
        total_loss_G = percep_loss

    # (LS)GAN loss and feature matching loss
    if loss_flags.use_gan_loss:
        logger.info("Use GAN Loss.")
        with nn.parameter_scope("discriminator"):
            discriminator_maps_generated = multiscale_discriminator(pyramide_fake,
                                                                    kp=unlink_all(
                                                                        kp_driving),
                                                                    **model_params.discriminator_params,
                                                                    **model_params.common_params,
                                                                    test=test, comm=comm)

            discriminator_maps_real = multiscale_discriminator(pyramide_real,
                                                               kp=unlink_all(
                                                                   kp_driving),
                                                               **model_params.discriminator_params,
                                                               **model_params.common_params,
                                                               test=test, comm=comm)

        for v in discriminator_maps_generated["feature_maps_1"]:
            v.persistent = True
        discriminator_maps_generated["prediction_map_1"].persistent = True

        for v in discriminator_maps_real["feature_maps_1"]:
            v.persistent = True
        discriminator_maps_real["prediction_map_1"].persistent = True

        for i, scale in enumerate(model_params.discriminator_params.scales):
            key = f'prediction_map_{scale}'.replace('.', '-')
            lsgan_loss_weight = train_params.loss_weights.generator_gan
            # LSGAN loss for Generator
            if i == 0:
                gan_loss_gen = lsgan_loss(discriminator_maps_generated[key],
                                          lsgan_loss_weight)
            else:
                gan_loss_gen += lsgan_loss(discriminator_maps_generated[key],
                                           lsgan_loss_weight)
            # LSGAN loss for Discriminator
            if i == 0:
                gan_loss_dis = lsgan_loss(discriminator_maps_real[key],
                                          lsgan_loss_weight,
                                          discriminator_maps_generated[key])
            else:
                gan_loss_dis += lsgan_loss(discriminator_maps_real[key],
                                           lsgan_loss_weight,
                                           discriminator_maps_generated[key])
        gan_loss_dis.persistent = True
        loss_var_dict['gan_loss_dis'] = gan_loss_dis
        total_loss_D = gan_loss_dis
        total_loss_D.persistent = True

        gan_loss_gen.persistent = True
        loss_var_dict['gan_loss_gen'] = gan_loss_gen
        total_loss_G += gan_loss_gen

        if loss_flags.use_feature_matching_loss:
            logger.info("Use Feature Matching Loss.")
            fm_weights = train_params.loss_weights.feature_matching
            fm_loss = feature_matching_loss(discriminator_maps_real,
                                            discriminator_maps_generated,
                                            model_params,
                                            fm_weights)
            fm_loss.persistent = True
            loss_var_dict['feature_matching_loss'] = fm_loss
            total_loss_G += fm_loss

    # transform loss
    if loss_flags.use_equivariance_value_loss or loss_flags.use_equivariance_jacobian_loss:
        transform = Transform(bs, **config.train_params.transform_params)
        transformed_frame = transform.transform_frame(driving)

        with nn.parameter_scope("kp_detector"):
            transformed_kp = detect_keypoint(transformed_frame,
                                             **model_params.kp_detector_params,
                                             **model_params.common_params,
                                             test=test, comm=comm)
        persistent_all(transformed_kp)

        # Value loss part
        if loss_flags.use_equivariance_value_loss:
            logger.info("Use Equivariance Value Loss.")
            warped_kp_value = transform.warp_coordinates(
                transformed_kp['value'])
            eq_value_weight = train_params.loss_weights.equivariance_value

            eq_value_loss = equivariance_value_loss(kp_driving['value'],
                                                    warped_kp_value,
                                                    eq_value_weight)
            eq_value_loss.persistent = True
            loss_var_dict['equivariance_value_loss'] = eq_value_loss
            total_loss_G += eq_value_loss

        # jacobian loss part
        if loss_flags.use_equivariance_jacobian_loss:
            logger.info("Use Equivariance Jacobian Loss.")
            arithmetic_jacobian = transform.jacobian(transformed_kp['value'])
            eq_jac_weight = train_params.loss_weights.equivariance_jacobian
            eq_jac_loss = equivariance_jacobian_loss(kp_driving['jacobian'],
                                                     arithmetic_jacobian,
                                                     transformed_kp['jacobian'],
                                                     eq_jac_weight)
            eq_jac_loss.persistent = True
            loss_var_dict['equivariance_jacobian_loss'] = eq_jac_loss
            total_loss_G += eq_jac_loss

    assert total_loss_G is not None
    total_loss_G.persistent = True
    loss_var_dict['total_loss_gen'] = total_loss_G

    # -------------------- Create Monitors --------------------
    monitors_gen, monitors_dis, monitor_time, monitor_vis, log_dir = get_monitors(
        config, loss_flags, loss_var_dict)

    if device_id == 0:
        # Dump training info .yaml
        _ = shutil.copy(args.config, log_dir)  # copy the config yaml
        training_info_yaml = os.path.join(log_dir, "training_info.yaml")
        os.rename(os.path.join(log_dir, os.path.basename(args.config)),
                  training_info_yaml)
        # then add additional information
        with open(training_info_yaml, "a", encoding="utf-8") as f:
            f.write(f"\nlog_dir: {log_dir}\nsaved_parameter: None")

    # -------------------- Solver Setup --------------------
    solvers = setup_solvers(train_params)
    solver_generator = solvers["generator"]
    solver_discriminator = solvers["discriminator"]
    solver_kp_detector = solvers["kp_detector"]

    # max epochs
    num_epochs = train_params['num_epochs']

    # iteration per epoch
    num_iter_per_epoch = data_iterator.size // bs
    # will be increased by num_repeat
    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        num_iter_per_epoch *= config.train_params.num_repeats

    # modify learning rate if current epoch exceeds the number defined in
    lr_decay_at_epochs = train_params['epoch_milestones']  # ex. [60, 90]
    gamma = 0.1  # decay rate

    # -------------------- For finetuning ---------------------
    if args.ft_params:
        assert os.path.isfile(args.ft_params)
        logger.info(f"load {args.ft_params} for finetuning.")
        nn.load_parameters(args.ft_params)
        start_epoch = int(os.path.splitext(
            os.path.basename(args.ft_params))[0].split("epoch_")[1])

        # set solver's state
        for name, solver in solvers.items():
            saved_states = os.path.join(os.path.dirname(
                args.ft_params), f"state_{name}_at_epoch_{start_epoch}.h5")
            solver.load_states(saved_states)

        start_epoch += 1
        logger.info(f"Resuming from epoch {start_epoch}.")

    logger.info(
        f"Start training. Total epoch: {num_epochs - start_epoch}, {num_iter_per_epoch * n_devices} iter/epoch.")

    for e in range(start_epoch, num_epochs):
        logger.info(f"Epoch: {e} / {num_epochs}.")
        data_iterator._reset()  # rewind the iterator at the beginning

        # learning rate scheduler
        if e in lr_decay_at_epochs:
            logger.info("Learning rate decayed.")
            learning_rate_decay(solvers, gamma=gamma)

        for i in range(num_iter_per_epoch):
            _driving, _source = data_iterator.next()
            source.d = _source
            driving.d = _driving

            # update generator and keypoint detector
            total_loss_G.forward()

            if device_id == 0:
                monitors_gen.add((e * num_iter_per_epoch + i) * n_devices)

            solver_generator.zero_grad()
            solver_kp_detector.zero_grad()

            callback = None
            if n_devices > 1:
                params = [x.grad for x in solver_generator.get_parameters().values()] + \
                         [x.grad for x in solver_kp_detector.get_parameters().values()]
                callback = comm.all_reduce_callback(params, 2 << 20)
            total_loss_G.backward(clear_buffer=True,
                                  communicator_callbacks=callback)

            solver_generator.update()
            solver_kp_detector.update()

            if loss_flags.use_gan_loss:
                # update discriminator

                total_loss_D.forward(clear_no_need_grad=True)
                if device_id == 0:
                    monitors_dis.add((e * num_iter_per_epoch + i) * n_devices)

                solver_discriminator.zero_grad()

                callback = None
                if n_devices > 1:
                    params = [
                        x.grad for x in solver_discriminator.get_parameters().values()]
                    callback = comm.all_reduce_callback(params, 2 << 20)
                total_loss_D.backward(clear_buffer=True,
                                      communicator_callbacks=callback)

                solver_discriminator.update()

            if device_id == 0:
                monitor_time.add((e * num_iter_per_epoch + i) * n_devices)

            if device_id == 0 and ((e * num_iter_per_epoch + i) * n_devices) % config.monitor_params.visualize_freq == 0:
                images_to_visualize = [source.d,
                                       driving.d,
                                       generated["prediction"].d]
                visuals = combine_images(images_to_visualize)
                monitor_vis.add((e * num_iter_per_epoch + i)
                                * n_devices, visuals)

        if device_id == 0:
            if e % train_params.checkpoint_freq == 0 or e == num_epochs - 1:
                save_parameters(e, log_dir, solvers)

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/vox-256.yaml', type=str)
    parser.add_argument('--context', '-c', default='cudnn',
                        type=str, choices=['cudnn', 'cpu'])
    parser.add_argument('--info', default=None, type=str)
    parser.add_argument('--ft-params', default=None, type=str)
    args = parser.parse_args()

    train(args)

    return


if __name__ == '__main__':
    main()
