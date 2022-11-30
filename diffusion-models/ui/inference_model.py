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
from typing import Union, List

import numpy as np
import nnabla as nn
import gradio as gr

from config import load_saved_conf, LoadedConfig
from diffusion_model.model import Model
from dataset.common import resize_center_crop

from .left import LeftBlocks, LeftBlocksCreator

# t5 encording
T5_MODEL = "t5-11b"
t5_tokenizer = None
t5_model = None

def t5_encode(sentence, device):
    import torch
    global t5_tokenizer, t5_model

    # load models if needed
    if t5_tokenizer is None:
        from transformers import T5Tokenizer
        t5_tokenizer = T5Tokenizer.from_pretrained(T5_MODEL)
    
    if t5_model is None:
        from transformers import T5EncoderModel
        t5_model = T5EncoderModel.from_pretrained(T5_MODEL)
        t5_model.half().to(f"cuda:{device}")

    tokens = t5_tokenizer(sentence,
                          padding=True,
                          truncation=True,
                          max_length=256,
                          return_length=True,
                          return_tensors="pt")

    outputs = t5_model(input_ids=tokens.input_ids.to(f"cuda:{device}"))

    emb_pt = outputs.last_hidden_state.detach()
    
    # length should be used if input text is a batch
    length = tokens.length.detach()

    return emb_pt.cpu()


def repeat(v, num):
    return nn.Variable.from_numpy_array([v for _ in range(num)])


def setup_model_and_kwargs(conf: LoadedConfig, 
                           respacing_step: int,
                           class_id: Union[int, list],
                           text: str,
                           lowres_image: np.ndarray,
                           lowres_noise_level: int,
                           i2i_strength: float,
                           num_gen: int):
        conf.diffusion.respacing_step = respacing_step
        if i2i_strength is not None:
            conf.diffusion.t_start = int(conf.diffusion.max_timesteps * (1.0 - i2i_strength))

        model = Model(diffusion_conf=conf.diffusion,
                      model_conf=conf.model)
        
        # set up model_kwargs
        model_kwargs = {}
        
        # set up class condition
        if conf.model.class_cond:
            if isinstance(class_id, int):
                model_kwargs["class_label"] = repeat(int(class_id), num_gen)
            else:
                assert isinstance(class_id, (list, tuple, np.ndarray))
                assert len(class_id) == num_gen
                model_kwargs["class_label"] = nn.Variable.from_numpy_array(class_id)
        
        # setup text condition
        if conf.model.text_cond:
            # get t5 embedding
            device = nn.get_current_context().device_id
            t5_emb = t5_encode(text, device) # (1, len, emb_dim)

            # trauncate and padding
            mtl = conf.model.max_text_length
            t5_emb = t5_emb[0, :mtl].detach().cpu() # (len, emb_dim)
            t5_emb = np.pad(t5_emb, ((0, mtl - t5_emb.shape[0]), (0, 0))) # (max_len, emb_dim)

            model_kwargs["text_emb"] = repeat(t5_emb, num_gen)
        
        # set up low-res input
        if conf.model.low_res_size is not None:
            assert lowres_image is not None
            lowres_image = nn.Variable.from_numpy_array(lowres_image)

            if conf.model.noisy_low_res:
                assert isinstance(lowres_image, nn.Variable)
                s = repeat(float(lowres_noise_level), num_gen)
                lowres_image, aug_level = model.gaussian_conditioning_augmentation(
                    lowres_image, s)
                model_kwargs["input_cond_aug_timestep"] = aug_level

            model_kwargs["input_cond"] = lowres_image

        return model, model_kwargs


def _rescale_img(x: np.ndarray):
    return ((x + 1) * 127.5).astype(np.uint8)


UNIQUE_MODEL_NAMES=set()

class InferenceModel(object):
    def __init__(self,
                 conf: Union[str, LoadedConfig],
                 h5: str,
                 name: str) -> None:

        # model config
        # if a path is provided as conf, load and create LoadedConfig instance
        if isinstance(conf, str):
            conf = load_saved_conf(conf)
        
        self.conf: LoadedConfig = conf

        # model parameter
        # only support a path for h5 file as `h5`
        assert os.path.exists(h5), \
            f"h5 file '{h5}' for base model doesn't exist."

        # load parameters under `name` scope
        assert name not in UNIQUE_MODEL_NAMES, \
            f"Model name `{name}` is already used. Please specify an unique name for each InferenceModel."
        self.name = name
        UNIQUE_MODEL_NAMES.add(name)
        with nn.parameter_scope(name):
            nn.load_parameters(h5)

        self.eps_list_base = None
        self.noise_base = None

        self.input_blocks = None
        self.output_block = None

    def create_input_blocks(self,
                            class_id_block=None,
                            text_block=None):
        assert self.input_blocks is None, \
            "create_input_blocks() is called twice."

        self.input_blocks = LeftBlocksCreator(self.conf, 
                                              self.name,
                                              show_class_id=class_id_block is None,
                                              show_text=text_block is None).create()

        # replace some blocks with given blocks
        if class_id_block is not None:
            self.input_blocks.class_id = class_id_block
        
        if text_block is not None:
            self.input_blocks.text = text_block
        
        return self.input_blocks

    def create_output_blocks(self):
        assert self.output_block is None, \
            "create_output_blocks() is called twice."
        
        # todo: output block as a module
        self.output_block = gr.Gallery(label=f"{self.name} generated images")

    @staticmethod
    def _get_or_none(gradio_inputs, block):
        if block is None:
            return None
        
        assert block in gradio_inputs

        return gradio_inputs[block]


    def infer(self, 
              gradio_inputs,
              output_from_parent):
        gi = gradio_inputs
        b = self.input_blocks
        num_gen = 4 # todo
        sampler = self._get_or_none(gi, b.sampler)
        cf_w = self._get_or_none(gi, b.classifier_free_guidance_weight)
        x_init = self._get_or_none(gi, b.source_image)

        setup_args = {
            "conf": self.conf,
            "respacing_step": self._get_or_none(gi, b.respacing_step),
            "class_id": self._get_or_none(gi, b.class_id),
            "text": self._get_or_none(gi, b.text),
            "lowres_image": output_from_parent,
            "lowres_noise_level": self._get_or_none(gi, b.lowres_noise_level),
            "i2i_strength": self._get_or_none(gi, b.i2i_strength),
            "num_gen": num_gen
        }
        if x_init is None:
            # Force i2i_strength=0 if a source image is not given.
            setup_args["i2i_strength"] = 0

        model, model_kwargs \
            = setup_model_and_kwargs(**setup_args)
        if x_init is not None:
            x_init = np.tile(x_init.reshape((1,) + x_init.shape), (num_gen, ) + (1, ) * len(x_init.shape))
            x_init = x_init / 127.5 - 1
            if not self.conf.model.channel_last:
                x_init = np.transpose(x_init, (2, 0, 1))
            T_var = nn.Variable(shape=(num_gen, ))
            T_var.data.fill(model.diffusion.num_timesteps - 1)
            with nn.auto_forward():
                x_init = model.diffusion.q_sample(nn.Variable.from_numpy_array(x_init), T_var)

        with nn.parameter_scope(self.name):
            gen, _, _ = model.sample(shape=(num_gen, ) + self.conf.model.image_shape,
                                     x_init=x_init,
                                     model_kwargs=model_kwargs,
                                     use_ema=True,
                                     sampler=sampler,
                                     progress=True,
                                     classifier_free_guidance_weight=cf_w)

        # make sure all pixel values are in [-1, 1]
        gen = np.clip(gen.d, -1, 1)

        return gen

    # properties used for a gradio event
    @property
    def inputs(self):
        return self.input_blocks.as_tuple()

    @property
    def outputs(self):
        return self.output_block

    @property
    def callback(self):
        # assume the input of callback is a dict of {gradio.Block: value}
        def callback(inp):
            outputs = {}

            out = self.infer(inp, output_from_parent=None)

            if self.output_block is not None:
                outputs[self.output_block] = _rescale_img(out)

            return outputs
        
        return callback



class SequentialInferenceModel(object):
    def __init__(self, models:List[InferenceModel]) -> None:
        self.models = models

    @property
    def inputs(self):
        inputs = set()
        for model in self.models:
            blocks: LeftBlocks = model.input_blocks
            inputs.update(blocks.as_tuple())
        
        return inputs

    @property
    def outputs(self):
        outputs = []
        for model in self.models:
            outputs.append(model.output_block)
        
        return outputs

    @property
    def callback(self):
        # assume the input of callback is a dict of {gradio.Block: value}
        def callback(inp):
            outputs = {}

            out = None
            for model in self.models:
                out = model.infer(inp, out)

                if model.output_block is not None:
                    outputs[model.output_block] = _rescale_img(out)
            
            return outputs
        
        return callback
