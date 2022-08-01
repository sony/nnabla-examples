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


from typing import Union
import os
from dataclasses import dataclass

import hydra
import nnabla as nn
from nnabla.logger import logger
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from neu.misc import init_nnabla
from neu.reporter import get_tiled_image, save_tiled_image
from nnabla.utils.image_utils import imsave, imread

import config
from model import Model


cs = ConfigStore.instance()
cs.store(name="base_config", node=config.GenScriptConfig)
config.register_configs()

@hydra.main(version_base=None,
            config_path="conf", 
            config_name="config_generate")
def main(conf: config.GenScriptConfig):
    assert conf.generate.x_start_path is not None, "generate.x_start_path must be supecified for sdedit.py."
    assert os.path.exists(conf.generate.x_start_path), f"{conf.generate.x_start_path} is not found."

    # load diffusion and model config
    loaded_conf: config.LoadedConfig = config.load_saved_conf(conf.generate.config)

    comm = init_nnabla(ext_name="cudnn", device_id=conf.runtime.device_id,
                       type_config=conf.runtime.type_config, random_pseed=True)
    
    if comm.n_procs > 1:
        raise ValueError("Currentely sdedit.py doesn't support multiGPU.")

    # update t_start if provided
    if conf.generate.t_start is not None:
        assert conf.generate.t_start < loaded_conf.diffusion.max_timesteps, \
            "t_start must be smaller than max_timesteps." \
            f"(t_start: {conf.generate.t_start}, T: {loaded_conf.diffusion.max_timesteps})"
        loaded_conf.diffusion.t_start = conf.generate.t_start

    model = Model(diffusion_conf=loaded_conf.diffusion,
                  model_conf=loaded_conf.model)

    # load parameters
    assert os.path.exists(
        conf.generate.h5), f"{conf.generate.h5} is not found. Please make sure the h5 file exists."
    nn.parameter.load_parameters(conf.generate.h5)

    # load image
    # todo: support multiple images
    if loaded_conf.model.channel_last:
        h, w = loaded_conf.model.image_shape[:-1]
    else:
        h, w = loaded_conf.model.image_shape[1:]  
    x0 = imread(conf.generate.x_start_path, size=(w, h), channel_first=not loaded_conf.model.channel_last)
    x0 = x0[np.newaxis]

    # setup output dir
    os.makedirs(conf.generate.output_dir, exist_ok=True)

    # dump configs       
    logger.info("===== script config =====")
    print(OmegaConf.to_yaml(conf))

    logger.info("===== loaded config =====")
    print(OmegaConf.to_yaml(loaded_conf))

    # Generate
    # todo: apply SDEdit several times
    sample_out, xts, x_starts = model.sample(shape=[1, ] + loaded_conf.model.image_shape,
                                             x_start=nn.Variable.from_numpy_array(
                                                    x0 / 127.5 - 1),
                                             dump_interval=1,
                                             use_ema=conf.generate.ema,
                                             progress=comm.rank == 0,
                                             use_ddim=conf.generate.ddim)

    # scale back to [0, 255]
    sample_out = (sample_out + 1) * 127.5
    save_path = os.path.join(
        conf.generate.output_dir, f"gen.png")
    imsave(save_path, sample_out[0].astype(
        np.uint8), channel_first=not loaded_conf.model.channel_last)

    # create video for x_starts
    if conf.generate.save_xstart:
        import moviepy.editor as mp
        clips = []
        for i in range(len(x_starts)):
            xstart = x_starts[i][1]
            assert isinstance(xstart, np.ndarray)
            im = get_tiled_image(
                np.clip((xstart + 1) * 127.5, 0, 255), channel_last=False).astype(np.uint8)
            clips.append(im)

        clip = mp.ImageSequenceClip(clips, fps=5)
        clip.write_videofile(os.path.join(
            conf.generate.output_dir, f"pred_x0_along_time.mp4"))


if __name__ == "__main__":
    main()
