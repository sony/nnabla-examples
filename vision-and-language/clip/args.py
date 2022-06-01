# Copyright 2022 Sony Group Corporation.
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


def get_args(monitor_path="tmp.monitor"):
    import argparse
    import os
    description = ("NNabla implementation of CLIP")
    parser = argparse.ArgumentParser(description)
    parser.add_argument('--context', '-c', type=str,
                        default=None, help="Extension modules. ex) 'cpu', 'cudnn'.")
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type configuration (float or half)')
    parser.add_argument("--batch-size", "-b", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--monitor-path", "-m",
                        type=str, default=monitor_path,
                        help='Path monitoring logs saved.')
    parser.add_argument("--epochs", "-ep", type=int, default=-1,
                        help='Number of training epochs.')
    parser.add_argument("--max-iter", "-i", type=int, default=-1,
                        help='Max iterations.')
    parser.add_argument("--warmup-iter", "-wi", type=int, default=10000,
                        help='iterations for warmup.')
    parser.add_argument("--device-id", "-d", type=int, default=0,
                        help='Device ID the training run on. This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--model-save-interval",
                        type=float, default=5000,
                        help='Model save interval')
    parser.add_argument("--model-save-path", "-o",
                        type=str, default=monitor_path,
                        help='Path where model parameters are saved.')
    parser.add_argument("--model-load-path",
                        type=str, default="asset/ViT-B-32.h5",
                        help='Path where model parameters are loaded.')
    parser.add_argument("--solver",
                        type=str,
                        help='Solvertype. [Adam, AdamW]')
    parser.add_argument('--lr', default=5e-4, type=float,
                        help="Learning rate")
    parser.add_argument('--beta1', default=0.9, type=float,
                        help="Beta1 of Adam")
    parser.add_argument('--beta2', default=0.98, type=float,
                        help="Beta2 of Adam")
    parser.add_argument('--eps', default=1.0e-6, type=float,
                        help="epsiron decay of Adam")
    parser.add_argument('--wd', default=1.0e-4, type=float,
                        help="weight decay of AdamW")
    parser.add_argument("--seed", default=412, type=int,
                        help='Use style encodoing in the other domain.')
    parser.add_argument("--finetuning", "-ft", action="store_true",
                        help='Finetune the pretrained model')
    parser.add_argument("--context_length", default=77, type=int,
                        help='Set context_length')
    parser.add_argument("--train-txt-path", default="asset/minisample.txt",
                        type=str,
                        help='Path to train txt file of image-text pair')
    parser.add_argument("--val-txt-path", default="asset/minisample.txt",
                        type=str,
                        help='Path to val txt file of image-text pair')
    parser.add_argument("--aggregate", "-ag", action="store_true",
                        help='Aggregate features across multiple gpus ')

    args = parser.parse_args()
    return args


def save_args(args, mode="train"):
    from nnabla import logger
    import os
    if not os.path.exists(args.monitor_path):
        os.makedirs(args.monitor_path)

    path = "{}/Arguments-{}.txt".format(args.monitor_path, mode)
    logger.info("Arguments are saved to {}.".format(path))
    with open(path, "w") as fp:
        for k, v in sorted(vars(args).items()):
            logger.info("{}={}".format(k, v))
            fp.write("{}={}\n".format(k, v))
