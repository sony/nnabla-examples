# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nnabla_ext.cuda.experimental import dali_iterator

from normalize_config import _pixel_mean, _pixel_std


class TrainPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, shard_id, image_dir, file_list, nvjpeg_padding,
                 prefetch_queue=3, seed=1, num_shards=1, channel_last=True, dtype="half"):
        super(TrainPipeline, self).__init__(
            batch_size, num_threads, shard_id, seed=seed, prefetch_queue_depth=prefetch_queue)
        self.input = ops.FileReader(file_root=image_dir, file_list=file_list,
                                    random_shuffle=True, num_shards=num_shards, shard_id=shard_id)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB,
                                        device_memory_padding=nvjpeg_padding,
                                        host_memory_padding=nvjpeg_padding)

        self.rrc = ops.RandomResizedCrop(device="gpu", size=(224, 224))
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT16 if dtype == "half" else types.FLOAT,
                                            output_layout=types.NHWC if channel_last else types.NCHW,
                                            crop=(224, 224),
                                            image_type=types.RGB,
                                            mean=_pixel_mean,
                                            std=_pixel_std,
                                            pad_output=True)
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.rrc(images)
        images = self.cmnp(images, mirror=self.coin())
        return images, labels.gpu()


class ValPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, shard_id, image_dir, file_list, nvjpeg_padding, seed=1, num_shards=1, channel_last=True, dtype='half'):
        super(ValPipeline, self).__init__(
            batch_size, num_threads, shard_id, seed=seed)
        self.input = ops.FileReader(file_root=image_dir, file_list=file_list,
                                    random_shuffle=False, num_shards=num_shards, shard_id=shard_id)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB,
                                        device_memory_padding=nvjpeg_padding,
                                        host_memory_padding=nvjpeg_padding)
        self.res = ops.Resize(device="gpu", resize_shorter=256)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT16 if dtype == "half" else types.FLOAT,
                                            output_layout=types.NHWC if channel_last else types.NCHW,
                                            crop=(224, 224),
                                            image_type=types.RGB,
                                            mean=_pixel_mean,
                                            std=_pixel_std,
                                            pad_output=True)

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.res(images)
        images = self.cmnp(images)
        return images, labels.gpu()


def get_data_iterators(args, comm, stream_event_handler):
    '''
    Creates and returns DALI data iterators for both datasets of training and
    validation.

    The datasets are partitioned in distributed training
    mode according to comm rank and number of processes.
    '''

    # Pipelines and Iterators for training
    train_pipe = TrainPipeline(args.batch_size, args.dali_num_threads, comm.rank,
                               args.train_dir,
                               args.train_list, args.dali_nvjpeg_memory_padding,
                               seed=comm.rank + 1,
                               num_shards=comm.n_procs,
                               channel_last=args.channel_last,
                               dtype=args.type_config)

    data = dali_iterator.DaliIterator(train_pipe)
    data.size = train_pipe.epoch_size("Reader") // comm.n_procs

    # Pipelines and Iterators for validation
    val_pipe = ValPipeline(args.batch_size, args.dali_num_threads, comm.rank,
                           args.val_dir, args.val_list, args.dali_nvjpeg_memory_padding,
                           seed=comm.rank + 1,
                           num_shards=comm.n_procs,
                           channel_last=args.channel_last,
                           dtype=args.type_config)
    vdata = dali_iterator.DaliIterator(val_pipe)
    vdata.size = val_pipe.epoch_size("Reader") // comm.n_procs

    return data, vdata
