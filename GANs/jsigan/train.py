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

import datetime
import os
import nnabla as nn
import nnabla.functions as F
import nnabla.solvers as S
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
from nnabla.ext_utils import get_extension_context
from args import get_config
from ops import model, gan_model
from data_loader import jsi_iterator
from utils import compute_psnr, get_learning_rate, CommunicatorWrapper


def setup_monitor(conf, monitor):
    """
    Setup monitor to keep track of losses and times to log them
    """
    jsi_monitor = {'rec_loss': MonitorSeries('rec_loss', monitor, interval=conf.monitor_interval),
                   'psnr': MonitorSeries('psnr', monitor, interval=conf.monitor_interval),
                   'lr': MonitorSeries('learning_rate', monitor,
                                       interval=conf.monitor_interval),
                   'g_final_loss': MonitorSeries('g_final_loss', monitor,
                                                 interval=conf.monitor_interval),
                   'd_final_fm_loss': MonitorSeries('d_final_fm_loss', monitor,
                                                    interval=conf.monitor_interval),
                   'd_final_detail_loss': MonitorSeries('d_final_detail_loss', monitor,
                                                        interval=conf.monitor_interval),
                   'g_adv_loss': MonitorSeries('g_adv_loss', monitor,
                                               interval=conf.monitor_interval),
                   'g_detail_adv_loss': MonitorSeries('g_detail_adv_loss', monitor,
                                                      interval=conf.monitor_interval),
                   'fm_loss': MonitorSeries('fm_loss', monitor,
                                            interval=conf.monitor_interval),
                   'fm_detail_loss': MonitorSeries('fm_detail_loss', monitor,
                                                   interval=conf.monitor_interval),
                   'time': MonitorTimeElapsed("Training time per epoch", monitor,
                                              interval=conf.monitor_interval)}
    return jsi_monitor


def main():
    conf = get_config()
    extension_module = conf.nnabla_context.context
    ctx = get_extension_context(
        extension_module, device_id=conf.nnabla_context.device_id)
    comm = CommunicatorWrapper(ctx)
    nn.set_default_context(comm.ctx)
    print("#GPU Count: ", comm.n_procs)

    data_iterator_train = jsi_iterator(conf.batch_size, conf, train=True)
    if conf.scaling_factor == 1:
        d_t = nn.Variable((conf.batch_size, 80, 80, 3),
                          need_grad=True)
        l_t = nn.Variable((conf.batch_size, 80, 80, 3), need_grad=True)

    else:
        d_t = nn.Variable((conf.batch_size, 160 / conf.scaling_factor, 160 / conf.scaling_factor, 3),
                          need_grad=True)
        l_t = nn.Variable((conf.batch_size, 160, 160, 3), need_grad=True)

    if comm.n_procs > 1:
        data_iterator_train = data_iterator_train.slice(
            rng=None, num_of_slices=comm.n_procs, slice_pos=comm.rank)

    monitor_path = './nnmonitor' + \
        str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

    monitor = Monitor(monitor_path)
    jsi_monitor = setup_monitor(conf, monitor)

    with nn.parameter_scope("jsinet"):
        nn.load_parameters(conf.pre_trained_model)
        net = model(d_t, conf.scaling_factor)
        net.pred.persistent = True
    rec_loss = F.mean(F.squared_error(net.pred, l_t))
    rec_loss.persistent = True
    g_final_loss = rec_loss

    if conf.jsigan:
        net_gan = gan_model(l_t, net.pred, conf)
        d_final_fm_loss = net_gan.d_adv_loss
        d_final_fm_loss.persistent = True
        d_final_detail_loss = net_gan.d_detail_adv_loss
        d_final_detail_loss.persistent = True
        g_final_loss = conf.rec_lambda * rec_loss + conf.adv_lambda * (
                net_gan.g_adv_loss + net_gan.g_detail_adv_loss) + conf.fm_lambda * (
                               net_gan.fm_loss + net_gan.fm_detail_loss)
        g_final_loss.persistent = True

    max_iter = data_iterator_train._size // (conf.batch_size)
    if comm.rank == 0:
        print("max_iter", data_iterator_train._size, max_iter)

    iteration = 0
    if not conf.jsigan:
        start_epoch = 0
        end_epoch = conf.adv_weight_point
        lr = conf.learning_rate * comm.n_procs
    else:
        start_epoch = conf.adv_weight_point
        end_epoch = conf.epoch
        lr = conf.learning_rate * comm.n_procs
        w_d = conf.weight_decay * comm.n_procs

    # Set generator parameters
    with nn.parameter_scope("jsinet"):
        solver_jsinet = S.Adam(alpha=lr, beta1=0.9, beta2=0.999, eps=1e-08)
        solver_jsinet.set_parameters(nn.get_parameters())

    if conf.jsigan:
        solver_disc_fm = S.Adam(alpha=lr, beta1=0.9, beta2=0.999, eps=1e-08)
        solver_disc_detail = S.Adam(
            alpha=lr, beta1=0.9, beta2=0.999, eps=1e-08)
        with nn.parameter_scope("Discriminator_FM"):
            solver_disc_fm.set_parameters(nn.get_parameters())
        with nn.parameter_scope("Discriminator_Detail"):
            solver_disc_detail.set_parameters(nn.get_parameters())

    for epoch in range(start_epoch, end_epoch):
        for index in range(max_iter):
            d_t.d, l_t.d = data_iterator_train.next()

            if not conf.jsigan:
                # JSI-net -> Generator
                lr_stair_decay_points = [200, 225]
                lr_net = get_learning_rate(lr, iteration, lr_stair_decay_points,
                                           conf.lr_decreasing_factor)
                g_final_loss.forward(clear_no_need_grad=True)
                solver_jsinet.zero_grad()
                if comm.n_procs > 1:
                    all_reduce_callback = comm.get_all_reduce_callback()
                    g_final_loss.backward(clear_buffer=True,
                                          communicator_callbacks=all_reduce_callback)
                else:
                    g_final_loss.backward(clear_buffer=True)
                solver_jsinet.set_learning_rate(lr_net)
                solver_jsinet.update()
            else:
                # GAN part (discriminator + generator)
                lr_gan = lr if epoch < conf.gan_lr_linear_decay_point \
                    else lr * (end_epoch - epoch) / (end_epoch - conf.gan_lr_linear_decay_point)
                lr_gan = lr_gan * conf.gan_ratio

                net.pred.need_grad = False

                # Discriminator_FM
                solver_disc_fm.zero_grad()
                d_final_fm_loss.forward(clear_no_need_grad=True)
                if comm.n_procs > 1:
                    all_reduce_callback = comm.get_all_reduce_callback()
                    d_final_fm_loss.backward(clear_buffer=True,
                                             communicator_callbacks=all_reduce_callback)
                else:
                    d_final_fm_loss.backward(clear_buffer=True)
                solver_disc_fm.set_learning_rate(lr_gan)
                solver_disc_fm.weight_decay(w_d)
                solver_disc_fm.update()

                # Discriminator_Detail
                solver_disc_detail.zero_grad()
                d_final_detail_loss.forward(clear_no_need_grad=True)
                if comm.n_procs > 1:
                    all_reduce_callback = comm.get_all_reduce_callback()
                    d_final_detail_loss.backward(clear_buffer=True,
                                                 communicator_callbacks=all_reduce_callback)
                else:
                    d_final_detail_loss.backward(clear_buffer=True)
                solver_disc_detail.set_learning_rate(lr_gan)
                solver_disc_detail.weight_decay(w_d)
                solver_disc_detail.update()

                # Generator
                net.pred.need_grad = True
                solver_jsinet.zero_grad()
                g_final_loss.forward(clear_no_need_grad=True)
                if comm.n_procs > 1:
                    all_reduce_callback = comm.get_all_reduce_callback()
                    g_final_loss.backward(clear_buffer=True,
                                          communicator_callbacks=all_reduce_callback)
                else:
                    g_final_loss.backward(clear_buffer=True)
                solver_jsinet.set_learning_rate(lr_gan)
                solver_jsinet.update()

            iteration += 1
            if comm.rank == 0:
                train_psnr = compute_psnr(net.pred.d, l_t.d, 1.)
                jsi_monitor['psnr'].add(iteration, train_psnr)
                jsi_monitor['rec_loss'].add(iteration, rec_loss.d.copy())
                jsi_monitor['time'].add(iteration)

            if comm.rank == 0:
                if conf.jsigan:
                    jsi_monitor['g_final_loss'].add(
                        iteration, g_final_loss.d.copy())
                    jsi_monitor['g_adv_loss'].add(
                        iteration, net_gan.g_adv_loss.d.copy())
                    jsi_monitor['g_detail_adv_loss'].add(iteration,
                                                         net_gan.g_detail_adv_loss.d.copy())
                    jsi_monitor['d_final_fm_loss'].add(
                        iteration, d_final_fm_loss.d.copy())
                    jsi_monitor['d_final_detail_loss'].add(
                        iteration, d_final_detail_loss.d.copy())
                    jsi_monitor['fm_loss'].add(
                        iteration, net_gan.fm_loss.d.copy())
                    jsi_monitor['fm_detail_loss'].add(
                        iteration, net_gan.fm_detail_loss.d.copy())
                    jsi_monitor['lr'].add(iteration, lr_gan)

        if comm.rank == 0:
            if not os.path.exists(conf.output_dir):
                os.makedirs(conf.output_dir)
            with nn.parameter_scope("jsinet"):
                nn.save_parameters(os.path.join(
                    conf.output_dir, "model_param_%04d.h5" % epoch))


if __name__ == "__main__":
    main()
