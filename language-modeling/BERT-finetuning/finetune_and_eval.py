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

import os
import numpy as np
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.solvers as S
import six.moves.cPickle as pickle

from nnabla.ext_utils import get_extension_context
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.monitor import Monitor, MonitorSeries

from args import get_args
from model import BertForSequenceClassification

from external.tokenization_bert import BertTokenizer
from external.metrics import glue_compute_metrics as compute_metrics
from external.processors import glue_output_modes as output_modes
from external.processors import glue_processors as processors
from external.processors import glue_convert_examples_to_features as convert_examples_to_features


class BERTDataSource(DataSource):

    def _get_data(self, position):
        input_id = self._input_ids[self._indices[position]]
        attention_mask = self._attention_masks[self._indices[position]]
        token_type_id = self._token_type_ids[self._indices[position]]
        label = self._labels[self._indices[position]]

        return input_id, attention_mask, token_type_id, label

    def __init__(self, args, tokenizer, shuffle=False, rng=None, evaluate=False):
        super(BERTDataSource, self).__init__(shuffle=shuffle)

        if rng is None:
            self.rng = np.random.RandomState(313)
        else:
            self.rng = rng

        input_ids, attention_masks, token_type_ids, labels = load_and_cache_examples(
            args, tokenizer, evaluate=evaluate)
        self._input_ids = input_ids
        self._attention_masks = attention_masks
        self._token_type_ids = token_type_ids
        self._labels = labels
        self._size = self._labels.shape[0]
        self._variables = ('x', 'y', 'z', 'a')
        self.reset()

    def reset(self):
        self._indices = self.rng.permutation(
            self._size) if self._shuffle else np.arange(self._size)
        return super(BERTDataSource, self).reset()


def lr_linear(current_step, total_steps):

    lr = float((total_steps-current_step)/total_steps)
    return lr


def train(args, train_dataset, tokenizer):
    """ Train the model """
    # Load the pretrianed model
    nn.load_parameters(args.pretrained_model)
    # Drop final layer for task-specific fine-tuning
    nn.parameter.pop_parameter('affine_seq_class/affine/W')
    nn.parameter.pop_parameter('affine_seq_class/affine/b')

    train_dataloader = data_iterator(
        train_dataset, batch_size=args.train_batch_size)

    global_step = 0
    train_loss = 0.0
    model = BertForSequenceClassification()

    input_ids = nn.Variable((args.train_batch_size, args.max_seq_length))
    attention_mask = nn.Variable((args.train_batch_size, args.max_seq_length))
    token_type_ids = nn.Variable((args.train_batch_size, args.max_seq_length))
    labels = nn.Variable((args.train_batch_size, ))

    input_ids_eval = nn.Variable((args.eval_batch_size, args.max_seq_length))
    attention_mask_eval = nn.Variable(
        (args.eval_batch_size, args.max_seq_length))
    token_type_ids_eval = nn.Variable(
        (args.eval_batch_size, args.max_seq_length))
    labels_eval = nn.Variable((args.eval_batch_size, ))

    activation = F.gelu
    if args.activation == 'relu':
        activation = F.relu
    loss, _, train_error = model(args, input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids, labels=labels,
                                 num_labels=args.num_labels, vocab_size=args.vocab_size,
                                 num_embed_dim=args.num_embed_dim,
                                 num_pos_ids=args.num_position_ids,
                                 num_attention_layers=args.num_attention_layers,
                                 num_attention_embed_dim=args.num_attention_embed_dim,
                                 num_attention_heads=args.num_attention_heads,
                                 num_attention_dim_feedforward=args.num_attention_dim_feedforward,
                                 attention_activation=activation, pool_outmap=args.num_pool_outmap,
                                 embed_dropout_prob=args.embed_dropout,
                                 attention_dropout_prob=args.attention_dropout,
                                 dropout_prob=args.last_dropout, test=False)

    loss.persistent = True
    if args.solver == 'Adam':
        solver = S.Adam(args.learning_rate, eps=args.adam_epsilon)
    else:
        solver = S.AdamW(args.learning_rate, eps=args.adam_epsilon)
    solver.set_parameters(nn.get_parameters())

    monitor = Monitor(args.output_dir)
    monitor_loss = MonitorSeries(
        "Training Loss", monitor, interval=10)
    monitor_eloss = MonitorSeries(
        "Evaluation Loss", monitor, interval=10)
    monitor_train_error = MonitorSeries(
        "Training Error Rate", monitor, interval=10)
    monitor_lr = MonitorSeries(
        "learning Rate", monitor, interval=10)

    total_steps = train_dataloader.size // args.train_batch_size
    var_linear = total_steps * args.num_train_epochs
    var_warmup = total_steps * (args.num_train_epochs - 1)
    for epoch in range(args.num_train_epochs):
        logger.info("Starting Epoch %d out of %d",
                    epoch+1, args.num_train_epochs)
        for it in range(total_steps):
            batch = train_dataloader.next()
            input_ids.d = batch[0]
            attention_mask.d = batch[1]
            token_type_ids.d = batch[2]
            labels.d = batch[3]

            learning_rate_linear = lr_linear(global_step, var_linear)
            learning_rate = args.learning_rate * learning_rate_linear

            if epoch == 0:
                learning_rate = args.learning_rate * (global_step/total_steps)
            if epoch > 0:
                learning_rate_linear = lr_linear(
                    (global_step-total_steps), var_warmup)
                learning_rate = args.learning_rate * learning_rate_linear

            solver.zero_grad()
            nn.forward_all([loss, train_error], clear_no_need_grad=True)
            loss.backward(clear_buffer=True)
            solver.weight_decay(args.weight_decay)
            solver.clip_grad_by_norm(args.max_grad_norm)
            solver.set_learning_rate(learning_rate)
            solver.update()

            monitor_loss.add(
                (train_dataloader.size//args.train_batch_size)*epoch+it,
                loss.d.copy())
            monitor_train_error.add(
                (train_dataloader.size//args.train_batch_size)*epoch+it,
                train_error.d.copy())
            monitor_lr.add(global_step, learning_rate)
            global_step += 1
            train_loss += F.mean(loss.data)

        eval_task_names = (
            "mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
        eval_outputs_dirs = (args.output_dir, args.output_dir +
                             '-MM') if args.task_name == "mnli" else (args.output_dir,)

        results = {}
        for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
            print(eval_task)
            eval_dataset = BERTDataSource(
                args, tokenizer, evaluate=True, shuffle=False)
            if not os.path.exists(eval_output_dir):
                os.makedirs(eval_output_dir)

            eval_dataloader = data_iterator(
                eval_dataset, batch_size=args.eval_batch_size)
            total_eval_steps = eval_dataloader.size // args.eval_batch_size
            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            out_label_ids = None
            tmp_eval_loss, logits, eval_error = model(args, input_ids=input_ids_eval,
                                                      attention_mask=attention_mask_eval,
                                                      token_type_ids=token_type_ids_eval, labels=labels_eval,
                                                      num_labels=args.num_labels, vocab_size=args.vocab_size,
                                                      num_embed_dim=args.num_embed_dim,
                                                      num_pos_ids=args.num_position_ids,
                                                      num_attention_layers=args.num_attention_layers,
                                                      num_attention_embed_dim=args.num_attention_embed_dim,
                                                      num_attention_heads=args.num_attention_heads,
                                                      num_attention_dim_feedforward=args.num_attention_dim_feedforward,
                                                      attention_activation=activation, pool_outmap=args.num_pool_outmap,
                                                      embed_dropout_prob=args.embed_dropout,
                                                      attention_dropout_prob=args.attention_dropout,
                                                      dropout_prob=args.last_dropout, test=True)

            tmp_eval_loss.persistent = True
            eval_loss += F.mean(tmp_eval_loss)
            for it in range(total_eval_steps):
                print(it, "  ", total_eval_steps)
                batch_eval = eval_dataloader.next()
                input_ids_eval.d = batch_eval[0]
                attention_mask_eval.d = batch_eval[1]
                token_type_ids_eval.d = batch_eval[2]
                labels_eval.d = batch_eval[3]
                nb_eval_steps += 1
                eval_loss.forward()
                monitor_eloss.add(it, eval_loss.d.copy())

                if preds is None:
                    preds = logits.d.copy()
                    out_label_ids = labels_eval.d.copy()
                else:
                    preds = np.append(preds, logits.d.copy(), axis=0)

                    out_label_ids = np.append(
                        out_label_ids, labels_eval.d.copy(), axis=0)
            eval_loss = eval_loss.d / nb_eval_steps
            if args.output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            elif args.output_mode == "regression":
                preds = np.squeeze(preds)

            result = compute_metrics(eval_task, preds, out_label_ids)
            results.update(result)

            output_eval_file = os.path.join(
                eval_output_dir, "", "eval_results.txt")
            with open(output_eval_file, "a") as writer:
                logger.info("***** Evaluation results {} *****".format(""))
                for key in sorted(result.keys()):
                    logger.info("%d  %s = %s\n", epoch +
                                1, key, str(result[key]))
                    writer.write("%d %s = %s\n" %
                                 (epoch+1, key, str(result[key])))
                print("results", results)
    return results


def load_and_cache_examples(args, tokenizer, evaluate=False):
    """
    Load and save cache files 

    args:
        args: argparser
        tokenizer: BERT Tokenizer
        evaluate: evaluation or not

    returns:
        dataset from cache
    """

    processor = processors[args.task_name]()
    output_mode = output_modes[args.task_name]
    cached_features_file = os.path.join(args.data_dir, 'nbla_cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(args.task_name)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        with open(cached_features_file, 'rb') as f:
            features = pickle.load(f)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(
            args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_token=tokenizer.convert_tokens_to_ids(
                                                    [tokenizer.pad_token])[0],
                                                pad_token_segment_id=0,
                                                )
        logger.info("Saving features into cached file %s",
                    cached_features_file)
        with open(cached_features_file, 'wb') as f:
            pickle.dump(features, f)

    all_input_ids = np.array([f.input_ids for f in features])
    all_attention_mask = np.array([f.attention_mask for f in features])
    all_token_type_ids = np.array([f.token_type_ids for f in features])
    all_labels = np.array([f.label for f in features])

    dataset = [all_input_ids, all_attention_mask,
               all_token_type_ids, all_labels]
    return dataset


def main():

    args = get_args()
    logger.info("Running in %s" % args.context)
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    args.num_labels = len(label_list)

    tokenizer_class = BertTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else
                                                args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    logger.info("Training/evaluation parameters %s", args)

    train_dataset = BERTDataSource(args, tokenizer, shuffle=True)
    train(args, train_dataset, tokenizer)


if __name__ == "__main__":
    main()
