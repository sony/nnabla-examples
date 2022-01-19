# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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


def get_args(monitor_path='tmp.monitor', epochs=300, model_save_path='tmp.monitor', learning_rate=1e-3, batch_size=64, weight_decay=0, n_devices=4, warmup_epoch=5):
    """
    Get command line arguments.

    Arguments set the default values of command line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", "-b", type=int, default=batch_size,
                        help="Local batch size, e.g., batch size per worker.")
    parser.add_argument("--learning-rate", "-l",
                        type=float, default=learning_rate,
                        help=f'initial learning rate (default:{learning_rate})')
    parser.add_argument("--monitor-path", "-m",
                        type=str, default=monitor_path,
                        help=f'Directory where monitoring files are stored (default:{monitor_path})')
    parser.add_argument("--epochs", type=int, default=epochs,
                        help='number of epochs to train (default:300)')
    parser.add_argument("--weight-decay", "-w",
                        type=float, default=weight_decay,
                        help='Weight decay factor to gradients')
    parser.add_argument("--device-id", "-d", type=str, default='0',
                        help='Device ID the training run on. This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--type-config", "-t", default='float',
                        choices=['float', 'half', 'mixed_half'],
                        help='Type of computation. (default:float)')
    parser.add_argument("--warmup-epoch", "-e", type=int, default=warmup_epoch,
                        help=f'Increase learning rate linearly over warm-up period (default:{warmup_epoch})')
    parser.add_argument("--model-save-interval", "-s", type=int, default=5000,
                        help='Iteration interval for saving model parameters (default:1000)')
    parser.add_argument("--model-save-path", "-o",
                        type=str, default=model_save_path,
                        help=f'Directory the model parameters saved (default:{model_save_path})')
    parser.add_argument('--context', '-c', type=str, default='cudnn',
                        choices=['cudnn', 'cuda', 'cpu'],
                        help="Extension module (default:cudnn)")
    parser.add_argument("--net", "-n", default='cifar10_resnet23',
                        choices=['cifar10_resnet23', 'cifar100_resnet23'],
                        help="Neural network architecture type (default:cifar10_resnet23)")
    parser.add_argument("--with-all-reduce-callback", action='store_true',
                        help="Use all_reduce_callback API instead of all_reduce")
    parser.add_argument('--sync-bn', action='store_true',
                        help="Use Synchronized batch normalization.")
    parser.add_argument("--use-latest-checkpoint", action='store_true',
                        help='load/save latest checkpoint in model_save_path')
    return parser.parse_args()
