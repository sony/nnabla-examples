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

import gradio as gr

from config import load_saved_conf, InferenceServerConfig

from .left import create_left_column_from_config, LeftBlocks
from .inference import InferenceModel

def create_demo(conf: InferenceServerConfig) -> gr.Blocks:
    # base model
    base_conf = load_saved_conf(conf.base_conf_path)
    
    # up1
    up1_conf = None
    use_1st_upsampler = (conf.up1_conf_path is not None) and (conf.up1_h5_path is not None)
    if use_1st_upsampler:
        up1_conf = load_saved_conf(conf.up1_conf_path)

 
    # build inference model
    inference_model = InferenceModel(base_conf=base_conf,
                                     base_h5=conf.base_h5_path,
                                     up1_conf=up1_conf,
                                     up1_h5=conf.up1_h5_path)

    inputs = []
    with gr.Blocks() as demo:
        # user inputs on the left column
        with gr.Row():
            # base model
            with gr.Column():
                with gr.Tab("Base model configurations"):
                    base_blocks: LeftBlocks \
                        = create_left_column_from_config(base_conf, "base model")
                    
                    inputs += [base_blocks.respacing_step,
                            base_blocks.sampler,
                            base_blocks.class_id,
                            base_blocks.classifier_free_guidance_weight]

                # up1
                if use_1st_upsampler:
                    with gr.Tab("1st upsampler configuration"):
                        up1_blocks: LeftBlocks \
                            = create_left_column_from_config(up1_conf, "1st upsampler")
                        
                        inputs += [up1_blocks.respacing_step,
                                up1_blocks.sampler,
                                up1_blocks.class_id,
                                up1_blocks.classifier_free_guidance_weight,
                                up1_blocks.lowres_noise_level]

                btn = gr.Button("Generate")

            image_output = gr.Gallery(label="generated images")

        # create callback
        btn.click(inference_model.create_callback(),
                  inputs=inputs,
                  outputs=[image_output])
            

    return demo