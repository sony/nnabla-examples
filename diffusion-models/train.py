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


import fnmatch
import os
import queue
import sys

import hydra
import nnabla as nn
import nnabla.solvers as S
import numpy as np
from hydra.core.config_store import ConfigStore
from neu.checkpoint_util import load_checkpoint, save_checkpoint
from neu.misc import get_current_time, init_nnabla
from neu.mixed_precision import MixedPrecisionManager
from neu.reporter import KVReporter, save_tiled_image
from nnabla.logger import logger
from omegaconf import OmegaConf

import config
from dataset import get_dataset
from diffusion import is_learn_sigma
from layers import adaptive_pooling_2d
from model import Model
from utils import (create_ema_op, get_lr_scheduler, init_checkpoint_queue,
                   sum_grad_norm)


def refine_monitor_files(dir_path, start_iter):
    for filename in os.listdir(dir_path):
        if filename.endswith(("series.txt", "timer.txt")):
            contents = []
            with open(os.path.join(dir_path, filename), "r") as f:
                for line in f.readlines():
                    iter = line.strip().split(" ")[0]
                    if int(iter) >= start_iter:
                        break

                    contents.append(line)

            with open(os.path.join(dir_path, filename), "w") as f:
                f.write("".join(contents))


def setup_resume(output_dir, dataset, solvers, is_master=False):
    # return
    start_iter = 0
    new_output_dir = output_dir

    parent = os.path.dirname(os.path.abspath(output_dir))

    if not os.path.exists(parent):
        # no previous checkpoint, train from scratch.
        return start_iter, output_dir

    all_logs = sorted(fnmatch.filter(
        os.listdir(parent), "*{}*".format(dataset)))
    if len(all_logs):
        latest_dir = os.path.join(parent, all_logs[-1])
        checkpoints = sorted(fnmatch.filter(
            os.listdir(latest_dir), "checkpoint_*.json"))

        # extract iteration count and use it as key
        iter_cp = [(int(x.split(".")[0].split("_")[-1]), x)
                   for x in checkpoints]

        for iter, checkpoint in reversed(sorted(iter_cp)):
            try:
                cp_path = os.path.join(latest_dir, checkpoint)
                start_iter = load_checkpoint(cp_path, solvers)

                for sname, slv in solvers.items():
                    slv.zero_grad()

                if is_master:
                    refine_monitor_files(latest_dir, start_iter)
                    init_checkpoint_queue(latest_dir)

                new_output_dir = latest_dir
                logger.info(f"Load checkpoint from {cp_path}")

                break
            except:
                logger.warning(
                    f"{checkpoint} is broken. Try to load previous checkpoints.")
        else:
            logger.warning("No valid checkpoint. Train from scratch")

    return start_iter, new_output_dir


def get_output_dir_name(org, dataset):
    return os.path.join(org, f"{get_current_time()}_{dataset}")


def augmentation(x, channel_last, random_flip=True):
    import nnabla.functions as F
    aug = x

    if random_flip:
        aug = F.random_flip(aug, axes=[2, ] if channel_last else None)

    return aug


# config
cs = ConfigStore.instance()
cs.store(name="base_config", node=config.TrainScriptConfig)

config.register_configs()


def create_gen_config(conf: config.TrainScriptConfig,
                      respacing_step: int) -> config.GenScriptConfig:
    conf_gen: config.GenScriptConfig \
         = OmegaConf.masked_copy(conf, ["diffusion", "model"])

    # setup respacing
    conf_gen.diffusion.respacing_step = respacing_step

    # disable dropout
    conf_gen.model.dropout = 0

    OmegaConf.set_readonly(conf_gen, True)

    return conf_gen


@hydra.main(version_base=None, config_path="conf", config_name="config_train")
def main(conf: config.TrainScriptConfig):
    # setup output dir
    conf.train.output_dir = get_output_dir_name(conf.train.output_dir,
                                                conf.dataset.name)

    # initialize nnabla runtime and get communicator
    comm = init_nnabla(ext_name="cudnn",
                       device_id=conf.runtime.device_id,
                       type_config=conf.runtime.type_config,
                       random_pseed=True)

    # create data iterator
    data_iterator = get_dataset(conf.dataset, comm)

    # build graph
    model = Model(diffusion_conf=conf.diffusion,
                  model_conf=conf.model)

    # setup input image
    # assume data_iterator returns [-1, 1]
    x = nn.Variable((conf.train.batch_size, ) + conf.model.image_shape)
    x_aug = augmentation(x, conf.model.channel_last, random_flip=True)

    # create low-resolution image
    model_kwargs = {}
    if conf.model.low_res_size is not None:
        assert len(conf.model.low_res_size) == 2
        x_low_res = adaptive_pooling_2d(x_aug,
                                        conf.model.low_res_size,
                                        mode="average",
                                        channel_last=conf.model.channel_last)

        # gaussian conditioning augmentation
        if conf.train.noisy_low_res:
            x_low_res, aug_level = model.gaussian_conditioning_augmentation(
                x_low_res)
            model_kwargs["input_cond_aug_timestep"] = aug_level

        # create model_kwargs
        model_kwargs["input_cond"] = x_low_res

    # setup class condition
    if conf.model.class_cond:
        model_kwargs["class_label"] = nn.Variable((conf.train.batch_size, ))
        model_kwargs["cond_drop_rate"] = conf.train.cond_drop_rate

    # setup text condition
    if conf.model.text_cond:
        assert conf.model.text_emb_shape is not None
        model_kwargs["text_emb"] = nn.Variable(
            (conf.train.batch_size, ) + conf.model.text_emb_shape)

    loss_dict, t = model.build_train_graph(x_aug,
                                           loss_scaling=None if conf.train.loss_scaling == 1.0 else conf.train.loss_scaling,
                                           model_kwargs=model_kwargs)
    assert loss_dict.batched_loss.shape == (conf.train.batch_size, )
    assert t.shape == (conf.train.batch_size, )

    # optimizer
    solver = S.Adam()
    solver.set_parameters(nn.get_parameters())

    lr_scheduler = get_lr_scheduler(conf.train)

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
    if conf.train.resume:
        # when resume, use the previous output_dir having the last checkpoint.
        start_iter, output_dir = setup_resume(conf.train.output_dir,
                                              conf.dataset.name,
                                              solvers, is_master=comm.rank == 0)
        conf.train.output_dir = output_dir

    image_dir = os.path.join(conf.train.output_dir, "image")
    if comm.rank == 0:
        os.makedirs(image_dir, exist_ok=True)

    comm.barrier()

    # Reporter
    reporter = KVReporter(comm, save_path=conf.train.output_dir,
                          skip_kv_to_monitor=False)
    # set all keys before to prevent synchronization error
    for i in range(4):
        reporter.set_key(f"loss_q{i}")
        if is_learn_sigma(conf.model.model_var_type):
            reporter.set_key(f"vlb_q{i}")

    if conf.train.progress:
        from tqdm import trange
        piter = trange(start_iter + 1, conf.train.n_iters + 1,
                       disable=comm.rank > 0, ncols=0)
    else:
        piter = range(start_iter + 1, conf.train.n_iters + 1)

    # freeze config to disable setting or updating values in conf.
    OmegaConf.resolve(conf)
    OmegaConf.set_readonly(conf, True)

    # setup model for generation
    conf_gen = create_gen_config(conf, respacing_step=4)
    gen_model = Model(diffusion_conf=conf_gen.diffusion,
                      model_conf=conf_gen.model)

    # dump config
    if comm.rank == 0:
        # show configs in stdout
        logger.info("===== configs =====")
        print(OmegaConf.to_yaml(conf))

        # save configs
        OmegaConf.save(conf, os.path.join(
            conf.train.output_dir, "config_train.yaml"))
        OmegaConf.save(conf_gen, os.path.join(
            conf.train.output_dir, "config_gen.yaml"))

    comm.barrier()

    mpm = MixedPrecisionManager(
        use_fp16=conf.runtime.type_config == "half",
        initial_log_loss_scale=10)

    # Queue to keep data instances for input_cond during sampling
    num_gen = 16
    data_queue = queue.Queue(maxsize=num_gen)

    for i in piter:
        # update learning rate
        if lr_scheduler is not None:
            cur_lr = lr_scheduler._get_lr(i, None)
        else:
            cur_lr = conf.train.lr

        # rescale lr to cancel backward accumulation
        solver.set_learning_rate(cur_lr / conf.train.accum)

        # evaluate graph
        dummy_solver_ema.zero_grad()  # just in case
        solver.zero_grad()

        retry_cnt = 0
        accum_cnt = 0
        while accum_cnt < conf.train.accum:
            batch = data_iterator.next()
            data = batch[0]
            x.d = data

            label = batch[1]  # if laion, this should be caption.

            text_emb = [None, ] * conf.train.batch_size
            if conf.model.text_cond:
                text_emb = batch[2]
                model_kwargs["text_emb"].d = text_emb

            if conf.model.class_cond:
                model_kwargs["class_label"].d = label

            # keep data for input_condition in generation step
            for data_idx in range(conf.train.batch_size):
                if data_queue.full():
                    break

                data_queue.put(
                    (data[data_idx], label[data_idx], text_emb[data_idx]))

            loss_dict.loss.forward(clear_no_need_grad=True)
            is_overflow = mpm.backward(loss_dict.loss, solver,
                                       clear_buffer=True)

            # Retry from the first accumulation step if overflow happens.
            if is_overflow:
                retry_cnt += 1

                # Raise if retry happens too many times.
                if retry_cnt == 100:
                    raise ValueError("Overflow happens too many times.")

                # mpm.backward resets grad of all params to zero in this case.
                accum_cnt = 0
                continue

            # fwd/bwd successes. Count up accum_cnt.
            accum_cnt += 1

            # reporting
            reporter.kv_mean("loss", loss_dict.loss)

            if is_learn_sigma(conf.model.model_var_type):
                reporter.kv_mean("vlb", loss_dict.vlb)

            # loss for each quantile
            for j in range(conf.train.batch_size):
                ti = t.d[j]
                q_level = int(ti) * 4 // conf.diffusion.max_timesteps
                assert q_level in (
                    0, 1, 2, 3), f"q_level should be one of [0, 1, 2, 3], but {q_level} is given."
                reporter.kv_mean(f"loss_q{q_level}", float(
                    loss_dict.batched_loss.d[j]))

                if is_learn_sigma(conf.model.model_var_type):
                    reporter.kv_mean(f"vlb_q{q_level}", loss_dict.vlb.d[j])

        # gradient accumulation
        assert accum_cnt == conf.train.accum

        # update
        is_overflow = mpm.update(solver, comm, clip_grad=conf.train.clip_grad)

        # check all processes not to have overflow.
        overflow_cnt = nn.NdArray.from_numpy_array(
            np.array([int(is_overflow)]))
        comm.all_reduce([overflow_cnt], division=False, inplace=True)

        if overflow_cnt.data > 0.5:
            # If even a single process has overflow, stop update and retry fwd/bwd again.

            # Basically overflow_cnt should be 0 or comm.n_procs
            # since allreduce over grads of all parmas has been performed above.
            assert comm.n_procs - int(overflow_cnt.data) < 1e-5, \
                "Some but not all nodes successfully update params without overflow. This is unintentional."

            continue

        # update ema params
        ema_op.forward(clear_no_need_grad=True)

        # grad norm
        if conf.train.dump_grad_norm:
            gnorm = sum_grad_norm(solver.get_parameters().values())
            reporter.kv_mean("grad", gnorm)

        # samples
        reporter.kv("samples", i * conf.train.batch_size *
                    comm.n_procs * conf.train.accum)

        # iteration (only for no-progress)
        if not conf.train.progress:
            reporter.kv("iteration", i)

        if i % conf.train.show_interval == 0:
            if conf.train.progress:
                desc = reporter.desc(
                    reset=True, sync=True if conf.runtime.type_config == "float" else False)
                piter.set_description(desc=desc)
            else:
                reporter.dump(file=sys.stdout if comm.rank ==
                              0 else None, reset=True, sync=True if conf.runtime.type_config == "float" else False)

            reporter.flush_monitor(i)

        if i % conf.train.save_interval == 0:
            if comm.rank == 0:
                save_checkpoint(conf.train.output_dir, i, solvers, n_keeps=3)

            comm.barrier()

        if conf.train.gen_interval > 0 and i % conf.train.gen_interval == 0:
            # sampling
            gen_model_kwargs = {}

            # setup condition vectors
            assert data_queue.qsize() == num_gen, \
                f"queue size is smaller than expected. ({data_queue.qsize()} < {num_gen})"
            image_list = []
            label_list = []
            text_emb_list = []
            for _ in range(num_gen):
                image, label, text_emb = data_queue.get()
                image_list.append(image)
                label_list.append(label)
                text_emb_list.append(text_emb)

            if conf.model.low_res_size is not None:
                # create lowres input from training dataset
                input_cond = nn.Variable.from_numpy_array(np.stack(image_list))

                input_cond_lowres = adaptive_pooling_2d(input_cond, conf.model.low_res_size,
                                                        mode="average",
                                                        channel_last=conf.model.channel_last)
                input_cond_lowres.forward(clear_buffer=True)

                gen_model_kwargs["input_cond"] = input_cond_lowres.get_unlinked_variable(
                    need_grad=False)
                gen_model_kwargs["input_cond_aug_timestep"] = nn.Variable.from_numpy_array(
                    np.zeros((num_gen, )))

            if conf.model.class_cond:
                gen_model_kwargs["class_label"] = nn.Variable.from_numpy_array(np.random.randint(low=0, high=conf.model.num_classes,
                                                                                                 size=(num_gen, )))
                # disable dropping condition for generation
                gen_model_kwargs["cond_drop_rate"] = 0

            if conf.model.text_cond:
                gen_model_kwargs["text_emb"] = nn.Variable.from_numpy_array(
                    np.asarray(text_emb_list))

                assert gen_model_kwargs["text_emb"].shape == (
                    num_gen, ) + model_kwargs["text_emb"].shape[1:]

            sample_out, _, _ = gen_model.sample(shape=(num_gen, ) + x.shape[1:],
                                                model_kwargs=gen_model_kwargs,
                                                use_ema=True, progress=False)
            assert sample_out.shape == (num_gen, ) + x.shape[1:]

            # scale back to [0, 255]
            sample_out = (sample_out + 1) * 127.5

            save_path = os.path.join(image_dir, f"gen_{i}_{comm.rank}.png")

            save_tiled_image(sample_out.astype(np.uint8),
                             save_path, channel_last=conf.model.channel_last)

            if conf.model.text_cond:
                with open(os.path.join(image_dir, f"gen_{i}_{comm.rank}_script.txt"), "w") as f:
                    f.write("\n".join(label_list))


if __name__ == "__main__":
    main()
