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

from dataclasses import dataclass, fields
from typing import List, Union
import gradio as gr

from config import LoadedConfig

@dataclass
class LeftBlocks():
    # inference configurations
    respacing_step: Union[None, gr.Slider] = None
    sampler: Union[None, gr.Slider] = None

    # for any types of conditional model
    classifier_free_guidance_weight: Union[None, gr.Dropdown] = None
    
    # for class conditional model
    class_id: Union[None, gr.Dropdown] = None

    # for text conditional model
    text: Union[None, gr.Textbox] = None

    # for upsampler
    lowres_noise_level: Union[None, gr.Dropdown] = None

    # for img2img
    source_image: Union[None, gr.Image] = None
    i2i_strength: Union[None, gr.Slider] = None

    def as_tuple(self):
        # we cannot use dataclasses.astuple,
        # because it returns deepcopy of each fields which have difference id.
        ret = set()
        for fld in fields(self):
            attr = getattr(self, fld.name)
            if attr is None:
                continue
            ret.add(attr)

        return ret

class LeftBlocksCreator(object):
    def __init__(self, 
                 conf: LoadedConfig,
                 name: str,
                 *,
                 show_class_id=True,
                 show_text=True):
        self.conf = conf
        self.name = name

        self.show_class_id = show_class_id
        self.show_text = show_text

    def _add_prefix(self, label):
        return f"[{self.name}] " + str(label)

    @property
    def respacing_step(self) -> gr.Slider:
        max_timestep = self.conf.diffusion.max_timesteps
        return gr.Slider(minimum=0,
                         maximum=max_timestep,
                         value=max_timestep // 50,  # default value
                         step=1,
                         label=self._add_prefix("Respacing step"))

    @property
    def sampler(self) -> gr.Dropdown:
        return gr.Dropdown(choices=["ddpm", "ddim", "plms", "dpm2"],
                           value="ddpm",
                           label=self._add_prefix("sampler"))

    @property
    def class_id(self) -> Union[None, gr.Dropdown]:
        if not self.conf.model.class_cond:
            return None
        
        # todo: show class name rather than id
        return gr.Dropdown(choices=list(range(0, self.conf.model.num_classes)),
                           value=0,
                           visible=self.show_class_id,
                           label=self._add_prefix("class id"))

    @property
    def text(self) -> Union[None, gr.Text]:
        if not self.conf.model.text_cond:
            return None

        return gr.Textbox(visible=self.show_text,
                          label=self._add_prefix("text input"))

    @property
    def cfguide(self) -> Union[None, gr.Slider]:
        if not (self.conf.model.class_cond or self.conf.model.text_cond):
            return None

        return gr.Slider(minimum=0,
                         maximum=100,
                         value=0.1,
                         step=0.1,
                         label=self._add_prefix("classifier-free guidance weight"))

    @property
    def lowres_noise_level(self) -> Union[None, gr.Slider]:
        if not self.conf.model.noisy_low_res:
            return None

        return gr.Slider(minimum=0,
                         maximum=self.conf.diffusion.max_timesteps,
                         value=0,
                         step=1,
                         label=self._add_prefix("noise timestep for low-resolution image"))

    @property
    def source_image(self) -> gr.Image:
        im_shape = tuple(self.conf.model.image_size)
        return gr.Image(shape=im_shape, label=self._add_prefix("source image for img2img"))

    @property
    def i2i_strength(self) -> gr.Slider:
        max_timestep = self.conf.diffusion.max_timesteps
        return gr.Slider(minimum=0,
                         maximum=1.0,
                         value=0.0,  # default value
                         step=1.0 / max_timestep,
                         label=self._add_prefix("source image strength for img2img"))


    def create(self):
        return LeftBlocks(respacing_step=self.respacing_step,
                          sampler=self.sampler,
                          class_id=self.class_id,
                          text=self.text,
                          classifier_free_guidance_weight=self.cfguide,
                          lowres_noise_level=self.lowres_noise_level,
                          source_image=self.source_image,
                          i2i_strength=self.i2i_strength)
