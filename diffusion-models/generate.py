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
import moviepy.editor as mp

import click
import nnabla as nn
from nnabla.logger import logger
import numpy as np
from nnabla.utils.image_utils import imsave
from neu.misc import AttrDict, init_nnabla
from neu.reporter import save_tiled_image, get_tiled_image
from neu.yaml_wrapper import read_yaml

from model import Model
from diffusion import ModelVarType


@click.command()
# configs for generating process
@click.option("--device-id", default='0', help="Device id.", show_default=True)
@click.option("--samples", default=16, help="# of generating samples on each device.", show_default=True)
@click.option("--batch-size", default=32, help="# of generating samples for each inference.", show_default=True)
# model configs
@click.option("--config", required=True, type=str, help="A path for config file.")
@click.option("--h5", required=True, type=str, help="A path for parameter to load.")
@click.option("--ema/--no-ema", default=True, help="Use ema params or not.")
@click.option("--ddim/--no-ddim", default=False, help="Use ddim sampler to generate data.", show_default=True)
@click.option("--sampling-interval", "-s", default=None, type=int, help="A timestep interval for sampling.")
# configs for dumping
@click.option("--output-dir", default="./outs", help="output dir", show_default=True)
@click.option("--tiled/--no-tiled", default=True, help="If true, generated images will be saved as tiled image.")
@click.option("--save-xstart/--no-save-xstart", default=False, help="If true, predicted xstarts at each timestep will be saved as video.")
def main(**kwargs):
    # set training args
    args = AttrDict(kwargs)

    assert os.path.exists(
        args.config), f"{args.config} is not found. Please make sure the config file exists."
    conf = read_yaml(args.config)

    comm = init_nnabla(ext_name="cudnn", device_id=args.device_id,
                       type_config="float", random_pseed=True)
    if args.sampling_interval is None:
        args.sampling_interval = 1

    use_timesteps = list(
        range(0, conf.num_diffusion_timesteps, args.sampling_interval))
    if use_timesteps[-1] != conf.num_diffusion_timesteps - 1:
        # The last step should be included always.
        use_timesteps.append(conf.num_diffusion_timesteps - 1)

    # setup model variance type
    model_var_type = ModelVarType.FIXED_SMALL
    if "model_var_type" in conf:
        model_var_type = ModelVarType.get_vartype_from_key(conf.model_var_type)

    model = Model(beta_strategy=conf.beta_strategy,
                  use_timesteps=use_timesteps,
                  model_var_type=model_var_type,
                  num_diffusion_timesteps=conf.num_diffusion_timesteps,
                  attention_num_heads=conf.num_attention_heads,
                  attention_resolutions=conf.attention_resolutions,
                  scale_shift_norm=conf.ssn,
                  base_channels=conf.base_channels,
                  channel_mult=conf.channel_mult,
                  num_res_blocks=conf.num_res_blocks)

    # load parameters
    assert os.path.exists(
        args.h5), f"{args.h5} is not found. Please make sure the h5 file exists."
    nn.parameter.load_parameters(args.h5)

    # Generate
    # sampling
    B = args.batch_size
    num_samples_per_iter = B * comm.n_procs
    num_iter = (args.samples + num_samples_per_iter -
                1) // num_samples_per_iter

    local_saved_cnt = 0
    for i in range(num_iter):
        logger.info(f"Generate samples {i + 1} / {num_iter}.")
        sample_out, _, x_starts = model.sample(shape=(B, ) + conf.image_shape[1:],
                                               dump_interval=1,
                                               use_ema=args.ema,
                                               progress=comm.rank == 0,
                                               use_ddim=args.ddim)

        # scale back to [0, 255]
        sample_out = (sample_out + 1) * 127.5

        if args.tiled:
            save_path = os.path.join(
                args.output_dir, f"gen_{local_saved_cnt}_{comm.rank}.png")
            save_tiled_image(sample_out.astype(np.uint8), save_path)
            local_saved_cnt += 1
        else:
            for b in range(B):
                save_path = os.path.join(
                    args.output_dir, f"gen_{local_saved_cnt}_{comm.rank}.png")
                imsave(save_path, sample_out[b].astype(
                    np.uint8), channel_first=True)
                local_saved_cnt += 1

        # create video for x_starts
        if args.save_xstart:
            clips = []
            for i in range(len(x_starts)):
                xstart = x_starts[i][1]
                assert isinstance(xstart, np.ndarray)
                im = get_tiled_image(
                    np.clip((xstart + 1) * 127.5, 0, 255), channel_last=False).astype(np.uint8)
                clips.append(im)

            clip = mp.ImageSequenceClip(clips, fps=5)
            clip.write_videofile(os.path.join(
                args.output_dir, f"pred_x0_along_time_{local_saved_cnt}_{comm.rank}.mp4"))


if __name__ == "__main__":
    main()
