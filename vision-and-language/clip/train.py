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


import os
import numpy as np
import argparse
import time
import datetime
import tqdm
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.communicators as C

from nnabla.monitor import Monitor, MonitorImage, MonitorSeries, MonitorTimeElapsed
from nnabla.ext_utils import get_extension_context
from nnabla.utils.profiler import GraphProfiler

from collections import OrderedDict

from args import get_args, save_args
from datasets import clip_data_iterator
from losses import get_logits
from scheduler import IterCosineLearningRateScheduler

import clip


def train(args):
    # Create Communicator and Context
    extension_module = "cudnn"
    ctx = get_extension_context(extension_module, type_config=args.type_config)
    comm = C.MultiProcessDataParalellCommunicator(ctx)
    comm.init()
    n_devices = comm.size
    mpi_rank = comm.rank
    mpi_local_rank = comm.local_rank
    device_id = mpi_local_rank
    ctx.device_id = str(device_id)
    nn.set_default_context(ctx)
    np.random.seed(args.seed)

    # Load checkpoint if file exists.
    if args.finetuning:
        # load nnabla params
        clip_model_path = args.model_load_path  # ex) 'asset/ViT-B-32.h5'
        clip.load(clip_model_path)
        print(f"Fine-tuning from {clip_model_path}.")
    else:
        # start training from initialized params
        clip_model_path = 'asset/ViT-B-32-initialized.h5'
        clip.load(clip_model_path)
        print(f"Training from {clip_model_path}.")

    # Create training graphs
    # Input
    b, c, h, w = args.batch_size, 3, args.image_size, args.image_size
    image = nn.Variable([b, c, h, w])
    text = nn.Variable([b, args.context_length])  # context_length = 77.

    # Graph
    image_features = clip.encode_image(image)
    text_features = clip.encode_text(text)
    image_features.persistent = True
    text_features.persistent = True

    if args.aggregate:
        assert comm is not None, "Set communicator"
        # gather all tensor across multiple gpu for a large matrix
        i_features_list = [nn.Variable(image_features.shape)
                           for _ in range(comm.size)]
        t_features_list = [nn.Variable(text_features.shape)
                           for _ in range(comm.size)]
        unliked_image_features = image_features.get_unlinked_variable()
        unliked_text_features = text_features.get_unlinked_variable()
        print(
            f"All Gather Executed: comm.size={comm.size}, comm.rank={comm.rank}, comm.local_rank={comm.local_rank}")
        comm.all_gather(unliked_image_features.data, [
                        i_features.data for i_features in i_features_list])
        comm.all_gather(unliked_text_features.data, [
                        t_features.data for t_features in t_features_list])

        # aggregate features_list to one features
        all_image_features = F.concatenate(
            unliked_image_features, *i_features_list[:comm.rank], *i_features_list[comm.rank+1:], axis=0)
        all_text_features = F.concatenate(
            unliked_text_features, *t_features_list[:comm.rank], *t_features_list[comm.rank+1:], axis=0)
        print(
            f"aggregated to image_features:{all_image_features.shape}, text_features:{all_text_features.shape}")
        logits_per_image, logits_per_text, logit_scale = get_logits(
            all_image_features, all_text_features, aggregate=args.aggregate)
        all_image_features.persistent = True
        all_text_features.persistent = True
    else:
        logits_per_image, logits_per_text, logit_scale = get_logits(
            image_features, text_features, aggregate=args.aggregate)
        logit_scale.persistent = True

    # Get ground_truth
    ground_truth_d = np.arange(
        logits_per_image.shape[0], dtype="int").reshape(-1, 1)
    ground_truth = nn.Variable((logits_per_image.shape[0], 1))

    # Cacl losses
    loss_image = F.mean(F.softmax_cross_entropy(
        logits_per_image, ground_truth))
    loss_image.persistent = True
    loss_text = F.mean(F.softmax_cross_entropy(logits_per_text, ground_truth))
    loss_text.persistent = True
    total_loss = (loss_image + loss_text) / 2
    total_loss.persistent = True

    # Create validation graphs
    # Input
    image_val = nn.Variable([b, c, h, w])
    # context_length = 77. Should be fixed following the official CLIP implement.
    text_val = nn.Variable([b, args.context_length])
    # Graph
    image_features_val = clip.encode_image(image_val)
    text_features_val = clip.encode_text(text_val)
    logits_per_image_val, logits_per_text_val, _ = get_logits(
        image_features_val, text_features_val, aggregate=args.aggregate)
    # Get ground_truth
    ground_truth_d_val = np.arange(
        logits_per_image_val.shape[0], dtype="int").reshape(-1, 1)
    ground_truth_val = nn.Variable((logits_per_image_val.shape[0], 1))
    # Culc losses
    loss_image_val = F.mean(F.softmax_cross_entropy(
        logits_per_image_val, ground_truth_val))
    loss_text_val = F.mean(F.softmax_cross_entropy(
        logits_per_text_val, ground_truth_val))
    total_loss_val = (loss_image_val + loss_text_val) / 2

    # Solver
    # Divide parameters into two groups mainly for AdamW, whose gain and bias are unsuitable for weight decay.
    def exclude(
        n, p): return p.ndim < 2 or "/bn" in n or "/ln" in n or "/b" in n or 'logit_scale' in n

    def include(n, p): return not exclude(n, p)

    named_parameters = nn.get_parameters()

    gain_or_bias_params = OrderedDict(
        [(n, p) for n, p in named_parameters.items() if exclude(n, p) and p.need_grad])
    rest_params = OrderedDict(
        [(n, p) for n, p in named_parameters.items() if include(n, p) and p.need_grad])

    if args.solver == "Adam":
        solver1 = S.Adam(args.lr, args.beta1, args.beta2, args.eps)
        solver2 = S.Adam(args.lr, args.beta1, args.beta2, args.eps)
    elif args.solver == "AdamW":
        solver1 = S.AdamW(args.lr, args.beta1, args.beta2, args.eps, 0.0)
        # ex) --lr 5.0e-4 --beta1 0.9 --beta2 0.98 --eps 1.0e-6 --wd 1.0e-4
        solver2 = S.AdamW(args.lr, args.beta1, args.beta2, args.eps, args.wd)
    else:
        raise NotImplementedError()
    solver1.set_parameters(gain_or_bias_params)
    solver2.set_parameters(rest_params)

    # Monitor
    monitor_interval = 100 * args.batch_size * n_devices
    monitor = Monitor(args.monitor_path)
    monitor_time = MonitorTimeElapsed(
        "Training time", monitor, interval=monitor_interval)
    monitor_loss_image = MonitorSeries(
        "Loss Image", monitor, interval=monitor_interval)
    monitor_loss_text = MonitorSeries(
        "Loss Text", monitor, interval=monitor_interval)
    monitor_total_loss = MonitorSeries(
        "Total Loss", monitor, interval=monitor_interval)
    monitor_logit_scale = MonitorSeries(
        "Logit Scale", monitor, interval=monitor_interval)
    monitor_losses = [
        (monitor_loss_image, loss_image),
        (monitor_loss_text, loss_text),
        (monitor_total_loss, total_loss),
        (monitor_logit_scale, logit_scale)
    ]
    # Monitor for eval
    monitor_loss_val = MonitorSeries(
        "Eval Loss", monitor, interval=1)  # monitoring each_epoch
    monitor_val_loss = [
        (monitor_loss_val, total_loss_val),
    ]

    # Data Iterator
    rng = np.random.RandomState(args.seed)
    train_loader = clip_data_iterator(
        clip, args.train_txt_path, args.batch_size, shuffle=True, rng=rng)
    print(f"train_loader.size: {train_loader.size}")
    if comm.size > 1:
        train_loader = train_loader.slice(
            rng=None, num_of_slices=comm.size, slice_pos=comm.rank)
        print(
            f"train_loader.size: {train_loader.size}, num_of_slices={comm.size}, slice_pos={comm.rank}")
    val_loader = clip_data_iterator(
        clip, args.val_txt_path, batch_size=args.batch_size, shuffle=False)
    iters_per_one_epoch = train_loader.size // args.batch_size

    comm.barrier()

    # set max iteration
    if args.max_iter == -1:
        args.max_iter = args.epochs * iters_per_one_epoch
        print(f"Max iteration = {args.max_iter}")

    # Learning rate scheduler
    learning_rate_scheduler = IterCosineLearningRateScheduler(
        base_lr=args.lr,
        iters=args.max_iter,
        warmup_iters=args.warmup_iter
    )

    comm.barrier()

    # Train loop
    for i in range(args.max_iter):
        ii = i * args.batch_size * n_devices
        current_epoch = i // iters_per_one_epoch

        image_data, text_data = train_loader.next()
        image.d, text.d = image_data, text_data
        ground_truth.d = ground_truth_d

        if args.aggregate:
            image_features.forward(clear_no_need_grad=True)
            text_features.forward(clear_no_need_grad=True)

            comm.all_gather(unliked_image_features.data, [
                            i_features.data for i_features in i_features_list])
            comm.all_gather(unliked_text_features.data, [
                            t_features.data for t_features in t_features_list])

            unliked_image_features.grad.zero()
            unliked_text_features.grad.zero()

        total_loss.forward(clear_no_need_grad=True)

        solver1.zero_grad()
        solver2.zero_grad()
        solver1.set_learning_rate(
            learning_rate_scheduler._get_lr(
                current_epoch=current_epoch, current_iter=i)
        )
        solver2.set_learning_rate(
            learning_rate_scheduler._get_lr(
                current_epoch=current_epoch, current_iter=i)
        )

        total_loss.backward(clear_buffer=True)

        if args.aggregate:
            image_features.backward(grad=None, clear_buffer=True)
            text_features.backward(grad=None, clear_buffer=True)
            comm.all_reduce([w.grad for w in gain_or_bias_params.values()])
            comm.all_reduce([w.grad for w in rest_params.values()])

        solver1.update()
        solver2.update()

        if mpi_local_rank == 0:
            # Monitor
            monitor_time.add(ii)
            for mon, loss in monitor_losses:
                mon.add(ii, loss.d)
            # Save Params and evaluate validation loss, at the end of each epoch
            if (i + 1) % iters_per_one_epoch == 0:
                # save
                nn.save_parameters(os.path.join(
                    args.monitor_path, "param_epoch_{:05d}.h5".format(current_epoch)))
                # val loss
                val_loss = []
                for vi in tqdm.tqdm(range(val_loader.size // args.batch_size)):
                    image_data_val, text_data_val = val_loader.next()
                    image_val.d, text_val.d = image_data_val, text_data_val
                    ground_truth_val.d = ground_truth_d_val
                    total_loss_val.forward(clear_buffer=True)
                    val_loss.append(total_loss_val.d)
                val_loss = np.array(val_loss).mean()
                for mon, loss in monitor_val_loss:
                    mon.add(current_epoch, val_loss)

        if (i + 1) % iters_per_one_epoch == 0:
            # re-shuffle for nnabla v1.27 or v1.28 by changing random seeds
            train_loader = clip_data_iterator(clip, args.train_txt_path, args.batch_size,
                                              shuffle=True, rng=np.random.RandomState(args.seed + (current_epoch+1)))
            if comm.size > 1:
                train_loader = train_loader.slice(
                    rng=None, num_of_slices=comm.size, slice_pos=comm.rank)

    if mpi_local_rank == 0:
        # Monitor
        for mon, loss in monitor_losses:
            mon.add(ii, loss.d)
        # Save
        nn.save_parameters(os.path.join(
            args.monitor_path, "param_iter_{:05d}.h5".format(i)))
        # val loss
        val_loss = []
        for vi in tqdm.tqdm(range(val_loader.size // args.batch_size)):
            image_data_val, text_data_val = val_loader.next()
            image_val.d, text_val.d = image_data_val, text_data_val
            ground_truth_val.d = ground_truth_d_val
            total_loss_val.forward(clear_buffer=True)
            val_loss.append(total_loss_val.d)
        val_loss = np.array(val_loss).mean()
        for mon, loss in monitor_val_loss:
            mon.add(current_epoch, val_loss)


def main():
    d = datetime.datetime.now()
    monitor_path = d.strftime(f"tmp.monitor/%Y-%m-%d-%H-%M-%S")
    args = get_args(monitor_path)
    save_args(args)
    train(args)


if __name__ == '__main__':
    main()
