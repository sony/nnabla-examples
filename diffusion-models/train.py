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
import sys

import fnmatch

import click
import nnabla as nn
import nnabla.solvers as S
from nnabla.logger import logger
import numpy as np

from neu.checkpoint_util import load_checkpoint, save_checkpoint
from neu.misc import AttrDict, get_current_time, init_nnabla
from neu.reporter import KVReporter, save_tiled_image
from neu.yaml_wrapper import write_yaml
from neu.mixed_precision import MixedPrecisionManager

from dataset import get_dataset
from model import Model
from utils import get_warmup_lr, sum_grad_norm, create_ema_op
from config import refine_args_by_dataset
from diffusion import ModelVarType, is_learn_sigma


def setup_resume(output_dir, dataset, solvers, is_master=False):
    # return
    start_iter = 0

    parent = os.path.dirname(os.path.abspath(output_dir))
    all_logs = sorted(fnmatch.filter(
        os.listdir(parent), "*{}*".format(dataset)))
    if len(all_logs):
        latest_dir = os.path.join(parent, all_logs[-1])
        checkpoints = sorted(fnmatch.filter(
            os.listdir(latest_dir), "checkpoint_*.json"))

        # extract iteration count and use it as key
        iter_cp = [(int(x.split(".")[0].split("_")[-1]), x) for x in checkpoints]

        for iter, checkpoint in reversed(sorted(iter_cp)):
            try:
                cp_path = os.path.join(latest_dir, checkpoint)
                start_iter = load_checkpoint(cp_path, solvers)

                for sname, slv in solvers.items():
                    slv.zero_grad()

                if is_master:
                    # copy loaded checkpoint to the current output_dir so that the current output_dir has at least one checkpoint
                    os.makedirs(output_dir, exist_ok=True)

                    import shutil
                    shutil.copy(cp_path, output_dir)

                logger.info(f"Load checkpoint from {cp_path}")

                break
            except:
                logger.warning(f"{checkpoint} is broken. Try to load previous checkpoints.")
        else:
            logger.warning("No valid checkpoint. Train from scratch")

    return start_iter


def get_output_dir_name(org, dataset):
    return os.path.join(org, f"{get_current_time()}_{dataset}")


def str_as_integer_list(ctx, param, value):
    if value is None or len(value) == 0 or value.lower() == "none":
        return None

    return tuple(int(x) for x in value.split(","))


@click.command()
# configs for training process
@click.option("--accum", default=1, help="# of gradient accumulation.", show_default=True)
@click.option("--type-config", default="float", type=str, help="Type configuration.", show_default=True)
@click.option("--device-id", default='0', help="Device id.", show_default=True)
@click.option("--batch-size", default=4, help="Batch size to train.", show_default=True)
@click.option("--n-iters", default=int(5e5), help="# of training iterations.", show_default=True)
@click.option("--progress/--no-progress", default=False, help="Use tqdm to show progress.")
@click.option("--resume/--no-resume", default=False, help="Resume training from the latest params saved at the same output_dir.")
@click.option("--loss-scaling", default=1.0, type=float, help="Loss scaling factor.", show_default=True)
@click.option("--lr", default=None, type=float, help="Learning rate.")
# model configs
@click.option("--beta-strategy", default="linear", help="Strategy to create betas.", show_default=True,
              type=click.Choice(["linear", "cosine"], case_sensitive=False), show_choices=True)
@click.option("--num-diffusion-timesteps", default=1000, help="Number of diffusion timesteps.", show_default=True)
@click.option("--ssn/--no-ssn", default=True, type=bool, help="use scale shift norm or not.")
@click.option("--resblock-resample/--no-resblock-resample", default=True, type=bool, help="Use resblock for down/up sampling.")
@click.option("--num-attention-heads", default=4, type=int, help="Number of multihead attention heads", show_default=True)
@click.option("--attention-resolutions", default="16,8", type=str, callback=str_as_integer_list,
              help="Resolutions which attention is applied. Comma separated string should be passed. If None, use default for dataset.")
@click.option("--num-attention-head-channels", default=None, type=int,
              help="Number of channels of each attention head."
                   "If specified, # of heads for each attention layer is automatically calculated and num-attention-heads will be ignored.", show_default=True)
@click.option("--base-channels", default=None, type=int, help="Base channel size. If None, use default for dataset.")
@click.option("--channel-mult", default=None, type=str, callback=str_as_integer_list,
              help="Channel multipliers for each block. Comma separated string should be passed. If None, use default for dataset.")
@click.option("--num-res-blocks", default=None, type=int, help="# of residual blocks. If None, use default for dataset.")
@click.option("--dropout", default=0.0, type=float, help="Dropout prob.", show_default=True)
@click.option("--model-var-type", type=click.Choice(ModelVarType.get_supported_keys(), case_sensitive=False),
              default="fixed_small", help="A type of the model variance.", show_default=True, show_choices=True)
@click.option("--channel-last/--no-channel-last", default=False, type=bool,
              help="Use channel last layout.", show_default=True)
# data related configs
@click.option("--dataset", default="custum", help="Dataset name to train model on.",
              type=click.Choice(["celebahq", "cifar10", "imagenet", "custom"], case_sensitive=False), show_choices=True)
@click.option("--image-size", default=None,
              type=int, help="Image size. Should be a integer and used for both height and width.")
@click.option("--data-dir", default="./data", help="The path for data directory.", show_default=True)
@click.option("--dataset-root-dir", default=None,
              help="The path for dataset root directory.", show_default=True)
@click.option("--dataset-on-memory/--no-dataset-on-memory", default=False,
              help="If True, the data once loaded will be kept on Host memory.", show_default=True)
@click.option("--fix-aspect-ratio/--no-fix-aspect-ratio", default=True,
              help="Whether keep aspect ratio or not for loaded training images.", show_default=True)
# configs for logging
@click.option("--output-dir", default="./logdir", help="output dir", show_default=True)
@click.option("--save-interval", default=int(1e4), help="Number of iters between saves.", show_default=True)
@click.option("--gen-interval", default=int(1e4), help="Number of iters between each generation.", show_default=True)
@click.option("--show-interval", default=10, help="Number of iters between showing current logging values.", show_default=True)
@click.option("--dump-grad-norm/--no-dump-grap-norm",
              default=False, help="Show sum of gradient norm of all params for each iteration.")
def main(**kwargs):
    # set training args
    args = AttrDict(kwargs)
    refine_args_by_dataset(args)

    args.output_dir = get_output_dir_name(args.output_dir, args.dataset)

    comm = init_nnabla(ext_name="cudnn", device_id=args.device_id,
                       type_config=args.type_config, random_pseed=True)

    data_iterator = get_dataset(args, comm)

    model = Model(beta_strategy=args.beta_strategy,
                  num_diffusion_timesteps=args.num_diffusion_timesteps,
                  model_var_type=ModelVarType.get_vartype_from_key(
                      args.model_var_type),
                  attention_num_heads=args.num_attention_heads,
                  attention_head_channels=args.num_attention_head_channels,
                  attention_resolutions=args.attention_resolutions,
                  scale_shift_norm=args.ssn,
                  base_channels=args.base_channels,
                  channel_mult=args.channel_mult,
                  num_res_blocks=args.num_res_blocks,
                  resblock_resample=args.resblock_resample,
                  channel_last=args.channel_last)

    use_timesteps = list(
        range(0, args.num_diffusion_timesteps, 4))  # sampling interval
    if use_timesteps[-1] != args.num_diffusion_timesteps - 1:
        # The last step should be included always.
        use_timesteps.append(args.num_diffusion_timesteps - 1)

    gen_model = Model(beta_strategy=args.beta_strategy,
                      use_timesteps=use_timesteps,
                      model_var_type=model.model_var_type,
                      num_diffusion_timesteps=args.num_diffusion_timesteps,
                      attention_num_heads=args.num_attention_heads,
                      attention_head_channels=args.num_attention_head_channels,
                      attention_resolutions=args.attention_resolutions,
                      scale_shift_norm=args.ssn,
                      base_channels=args.base_channels,
                      channel_mult=args.channel_mult,
                      num_res_blocks=args.num_res_blocks,
                      resblock_resample=args.resblock_resample,
                      channel_last=args.channel_last)

    # build graph
    x = nn.Variable(args.image_shape)  # assume data_iterator returns [0, 255]
    x_rescaled = x / 127.5 - 1  # rescale to [-1, 1]
    loss_dict, t = model.build_train_graph(x_rescaled,
                                           dropout=args.dropout,
                                           loss_scaling=None if args.loss_scaling == 1.0 else args.loss_scaling)
    assert loss_dict.batched_loss.shape == (args.batch_size, )
    assert t.shape == (args.batch_size, )

    # optimizer
    solver = S.Adam()
    solver.set_parameters(nn.get_parameters())

    # for ema update
    # Note: this should be defined after solver.set_parameters() to avoid update by solver.
    ema_op, ema_params = create_ema_op(nn.get_parameters(), 0.9999)
    dummy_solver_ema = S.Sgd()
    dummy_solver_ema.set_learning_rate(0)  # just in case
    dummy_solver_ema.set_parameters(ema_params)
    assert len(nn.get_parameters(grad_only=True)) == len(ema_params)
    assert len(nn.get_parameters(grad_only=False)) == 2 * len(ema_params)

    # for checkpoint
    solvers = {
        "main": solver,
        "ema": dummy_solver_ema,
    }

    start_iter = 0  # exclusive
    if args.resume:
        start_iter = setup_resume(args.output_dir, args.dataset, solvers, is_master=comm.rank == 0)

    if comm.rank == 0:
        image_dir = os.path.join(args.output_dir, "image")
        os.makedirs(image_dir, exist_ok=True)

    comm.barrier()

    # Reporter
    reporter = KVReporter(comm, save_path=args.output_dir,
                          skip_kv_to_monitor=False)
    # set all keys before to prevent synchronization error
    for i in range(4):
        reporter.set_key(f"loss_q{i}")
        if is_learn_sigma(model.model_var_type):
            reporter.set_key(f"vlb_q{i}")

    if args.progress:
        from tqdm import trange
        piter = trange(start_iter + 1, args.n_iters + 1,
                       disable=comm.rank > 0, ncols=0)
    else:
        piter = range(start_iter + 1, args.n_iters + 1)

    # dump config
    if comm.rank == 0:
        args.dump()
        write_yaml(os.path.join(args.output_dir, "config.yaml"), args)

    comm.barrier()

    mpm = MixedPrecisionManager(use_fp16=args.type_config == "half", initial_log_loss_scale=15)

    for i in piter:
        # update solver's lr
        # cur_lr = get_warmup_lr(lr, args.n_warmup, i)
        solver.set_learning_rate(args.lr)

        # evaluate graph
        dummy_solver_ema.zero_grad()  # just in case
        solver.zero_grad()

        retry_cnt = 0

        for accum_iter in range(args.accum):  # accumelate
            data, label = data_iterator.next()
            x.d = data.copy()

            loss_dict.loss.forward(clear_no_need_grad=True)

            all_reduce_cb = None
            # if accum_iter == args.accum - 1:
            #     all_reduce_cb = comm.get_all_reduce_callback(
            #         params=solver.get_parameters().values())

            is_overflow = mpm.backward(loss_dict.loss, solver,
                                       clear_buffer=True, communicator_callbacks=all_reduce_cb)

            if is_overflow:
                retry_cnt += 1
                if retry_cnt == 20:
                    raise ValueError("Overflow happens too many times.")

                accum_iter = 0
                continue

            if accum_iter == args.accum - 1:
                comm.all_reduce(
                    [x.grad for x in nn.get_parameters().values()], division=True, inplace=False)

            reporter.kv_mean("loss", loss_dict.loss)

            if is_learn_sigma(model.model_var_type):
                reporter.kv_mean("vlb", loss_dict.vlb)

            # loss for each quantile
            for j in range(args.batch_size):
                ti = t.d[j]
                q_level = int(ti) * 4 // args.num_diffusion_timesteps
                assert q_level in (
                    0, 1, 2, 3), f"q_level should be one of [0, 1, 2, 3], but {q_level} is given."
                reporter.kv_mean(f"loss_q{q_level}", float(
                    loss_dict.batched_loss.d[j]))

                if is_learn_sigma(model.model_var_type):
                    reporter.kv_mean(f"vlb_q{q_level}", loss_dict.vlb.d[j])

        # update
        is_overflow = mpm.update(solver, clip_grad=args.clip_grad)

        # update ema params
        if is_overflow:
            i -= 1
            continue
        else:
            ema_op.forward(clear_no_need_grad=True)

        # grad norm
        if args.dump_grad_norm:
            gnorm = sum_grad_norm(solver.get_parameters().values())
            reporter.kv_mean("grad", gnorm)

        # samples
        reporter.kv("samples", i * args.batch_size * comm.n_procs * args.accum)

        # iteration (only for no-progress)
        if not args.progress:
            reporter.kv("iteration", i)

        if i % args.show_interval == 0:
            if args.progress:
                desc = reporter.desc(
                    reset=True, sync=True if args.type_config == "float" else False)
                piter.set_description(desc=desc)
            else:
                reporter.dump(file=sys.stdout if comm.rank ==
                              0 else None, reset=True, sync=True if args.type_config == "float" else False)

            reporter.flush_monitor(i)

        if i > 0 and i % args.save_interval == 0:
            if comm.rank == 0:
                save_checkpoint(args.output_dir, i, solvers, n_keeps=3)

            comm.barrier()

        if i > 0 and args.gen_interval > 0 and i % args.gen_interval == 0:
            # sampling
            sample_out, _, _ = gen_model.sample(shape=(16, ) + x.shape[1:],
                                                use_ema=True, progress=False)
            assert sample_out.shape == (16, ) + args.image_shape[1:]

            # scale back to [0, 255]
            sample_out = (sample_out + 1) * 127.5

            save_path = os.path.join(image_dir, f"gen_{i}_{comm.rank}.png")

            save_tiled_image(sample_out.astype(np.uint8),
                             save_path, channel_last=args.channel_last)


if __name__ == "__main__":
    main()
