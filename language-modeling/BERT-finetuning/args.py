# Copyright 2021 Sony Group Corporation
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

import argparse


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--context', '-c', type=str,
                        default="cudnn", help="Extension path: cpu or cudnn.")
    parser.add_argument("--device-id", "-d", type=str, default='0',
                        help='device id (default: 0)')
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='computation precision. e.g. "float", "half".')
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--model_name_or_path",
                        default=None, type=str, required=True)
    parser.add_argument("--pretrained_model",
                        default="nbla_bert_params.h5", type=str)
    parser.add_argument("--task_name", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--num_labels", default=2, type=int)
    parser.add_argument("--vocab_size", default=30522, type=int)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--num_embed_dim", default=768, type=int)
    parser.add_argument("--num_position_ids", default=512, type=int)
    parser.add_argument("--num_attention_layers", default=12, type=int)
    parser.add_argument("--num_attention_embed_dim", default=768, type=int)
    parser.add_argument("--num_attention_heads", default=12, type=int)
    parser.add_argument("--num_attention_dim_feedforward",
                        default=3072, type=int)
    parser.add_argument("--num_pool_outmap", default=768, type=int)
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--solver", default="AdamW", type=str)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--activation", default="gelu",
                        type=str, choices=["gelu", "relu"],)
    parser.add_argument("--embed_dropout", default=0.1, type=float)
    parser.add_argument("--attention_dropout", default=0.1, type=float)
    parser.add_argument("--last_dropout", default=0.1, type=float)
    parser.add_argument("--num_train_epochs", default=3, type=int)
    parser.add_argument('--overwrite_output_dir', default=True)
    parser.add_argument('--overwrite_cache', action='store_true')

    args = parser.parse_args()

    return args
