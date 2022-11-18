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

from .inference_model import InferenceModel, SequentialInferenceModel


def create_demo(conf: InferenceServerConfig) -> gr.Blocks:
    models = []

    # base model
    base_model = InferenceModel(conf=conf.base_conf_path,
                                h5=conf.base_h5_path,
                                name="base")
    models.append(base_model)

    # up1
    use_1st_upsampler = (conf.up1_conf_path is not None) and (conf.up1_h5_path is not None)
    if use_1st_upsampler:
        up1_model = InferenceModel(conf=conf.up1_conf_path,
                                   h5=conf.up1_h5_path,
                                   name="1st upsampler")
        models.append(up1_model)


    inputs = []
    with gr.Blocks() as demo:
        with gr.Row():
            # left column for inputs
            with gr.Column():
                with gr.Tab("Base model configurations"):
                    base_model.create_input_blocks()

                # up1
                # initialize by empty block
                if use_1st_upsampler:
                    with gr.Tab("1st upsampler configuration"):
                        up1_model.create_input_blocks(class_id_block=base_model.input_blocks.class_id,
                                                      text_block=base_model.input_blocks.text)

                btn = gr.Button("Generate")

            # right column for outputs
            with gr.Column():
                base_model.create_output_blocks()
                
                if use_1st_upsampler:
                    up1_model.create_output_blocks()


        # create callback
        cb1 = SequentialInferenceModel(models)
        btn.click(cb1.callback,
                  inputs=cb1.inputs,
                  outputs=cb1.outputs)

    return demo
