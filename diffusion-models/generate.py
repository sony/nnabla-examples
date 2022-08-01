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

import hydra
import nnabla as nn
import numpy as np
from hydra.core.config_store import ConfigStore
from neu.misc import init_nnabla
from neu.reporter import get_tiled_image, save_tiled_image
from nnabla.logger import logger
from nnabla.utils.image_utils import imsave
from omegaconf import OmegaConf

import config
from dataset.common import SimpleDataIterator
from model import Model

cs = ConfigStore.instance()
cs.store(name="base_config", node=config.GenScriptConfig)
config.register_configs()

@hydra.main(version_base=None,
            config_path="conf", 
            config_name="config_generate")
def main(conf: config.GenScriptConfig):
    # load diffusion and model config
    loaded_conf: config.LoadedConfig = config.load_saved_conf(conf.generate.config)

    comm = init_nnabla(ext_name="cudnn", device_id=conf.runtime.device_id,
                       type_config=conf.runtime.type_config, random_pseed=True)
    
    # update respacing parameter based on given config
    loaded_conf.diffusion.respacing_step = conf.generate.respacing_step

    model = Model(diffusion_conf=loaded_conf.diffusion,
                  model_conf=loaded_conf.model)

    # load parameters
    assert os.path.exists(
        conf.generate.h5), f"{conf.generate.h5} is not found. Please make sure the h5 file exists."
    nn.parameter.load_parameters(conf.generate.h5)

    # Generate

    # setup data iterator for lowres samples
    B = conf.generate.batch_size




    # Generate
    # sampling
    B = args.batch_size
    num_samples_per_iter = B * comm.n_procs
    num_iter = (args.samples + num_samples_per_iter -
                1) // num_samples_per_iter

    # sampling
    local_saved_cnt = 0

    # setup output dir and show config
    if comm.rank == 0:
        os.makedirs(conf.generate.output_dir, exist_ok=True)        
        logger.info("===== script config =====")
        print(OmegaConf.to_yaml(conf))

        logger.info("===== loaded config =====")
        print(OmegaConf.to_yaml(loaded_conf))

    comm.barrier()

    for i in range(num_iter):
        logger.info(f"Generate samples {i + 1} / {num_iter}.")

        sample_out, xt_samples, x_starts = model.sample(shape=[B, ] + loaded_conf.model.image_shape,
                                                        noise=None,
                                                        dump_interval=-1,
                                                        use_ema=conf.generate.ema,
                                                        progress=comm.rank == 0,
                                                        use_ddim=conf.generate.ddim)

        # scale back to [0, 255]
        sample_out = (sample_out + 1) * 127.5

        if conf.generate.tiled:
            save_path = os.path.join(
                conf.generate.output_dir, f"gen_{local_saved_cnt}_{comm.rank}.png")
            save_tiled_image(sample_out.astype(np.uint8),
                             save_path,
                             channel_last=loaded_conf.model.channel_last)

            local_saved_cnt += 1
        else:
            for b in range(B):
                save_path = os.path.join(
                    conf.generate.output_dir, f"gen_{local_saved_cnt}_{comm.rank}.png")
                imsave(save_path, sample_out[b].astype(
                    np.uint8), channel_first=not loaded_conf.model.channel_last)

                local_saved_cnt += 1

        # create video for x_starts
        if conf.generate.save_xstart:
            import moviepy.editor as mp

            clips = []
            for i in range(len(x_starts)):
                xstart = x_starts[i][1]
                assert isinstance(xstart, np.ndarray)
                im = get_tiled_image(
                    np.clip((xstart + 1) * 127.5, 0, 255), channel_last=conf.channel_last).astype(np.uint8)
                clips.append(im)

            clip = mp.ImageSequenceClip(clips, fps=5)
            clip.write_videofile(os.path.join(
                conf.generate.output_dir, f"pred_x0_along_time_{local_saved_cnt}_{comm.rank}.mp4"))


if __name__ == "__main__":
    main()
