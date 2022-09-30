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

import hydra

from config import InferenceServerScriptConfig
from ui import create_demo

from neu.misc import init_nnabla


@hydra.main(version_base=None, 
            config_path="config/yaml",
            config_name="config_inference_server")
def main(conf: InferenceServerScriptConfig):
    # initialize nnabla context
    init_nnabla(ext_name="cudnn",
                device_id=conf.runtime.device_id,
                type_config=conf.runtime.type_config)

    # create demo
    demo = create_demo(conf.server)
    
    # launch server
    demo.launch(server_name="0.0.0.0", 
                server_port=conf.server.port)


if __name__ == "__main__":
    main()
