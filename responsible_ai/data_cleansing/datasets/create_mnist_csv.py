# Copyright 2021 Sony Corporation.
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

'''
Download MNIST examples as CSV.
'''
import os
import csv
import argparse
from imageio import imwrite
from contextlib import contextmanager
import numpy
import struct
import zlib
import tqdm

from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.data_source_loader import download
from .utils import get_filename_to_download, save_list_to_csv, split_data_into_train_val, ensure_dir


class MnistDataSource(DataSource):
    '''
    Get data directly from MNIST dataset from Internet(yann.lecun.com).
    '''

    def _get_data(self, position):
        image = self._images[self._indexes[position]]
        label = self._labels[self._indexes[position]]
        return (image, label)

    def __init__(self, train=True, shuffle=False, rng=None, output_dir=None):
        super(MnistDataSource, self).__init__(shuffle=shuffle)
        self._train = train
        if self._train:
            image_uri = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
            label_uri = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
        else:
            image_uri = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
            label_uri = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

        logger.info('Getting label data from {}.'.format(label_uri))
        # With python3 we can write this logic as following, but with
        # python2, gzip.object does not support file-like object and
        # urllib.request does not support 'with statement'.
        #
        #   with request.urlopen(label_uri) as r, gzip.open(r) as f:
        #       _, size = struct.unpack('>II', f.read(8))
        #       self._labels = numpy.frombuffer(f.read(), numpy.uint8).reshape(-1, 1)
        #
        label_output_file = get_filename_to_download(output_dir, label_uri)
        # file object returned
        r = download(label_uri, output_file=label_output_file)
        data = zlib.decompress(r.read(), zlib.MAX_WBITS | 32)
        _, size = struct.unpack('>II', data[0:8])
        self._labels = numpy.frombuffer(data[8:], numpy.uint8).reshape(-1, 1)
        r.close()
        logger.info('Getting label data done.')

        logger.info('Getting image data from {}.'.format(image_uri))
        image_output_file = get_filename_to_download(output_dir, image_uri)
        r = download(image_uri, output_file=image_output_file)
        data = zlib.decompress(r.read(), zlib.MAX_WBITS | 32)
        _, size, height, width = struct.unpack('>IIII', data[0:16])
        self._images = numpy.frombuffer(data[16:], numpy.uint8).reshape(
            size, 1, height, width)
        r.close()
        logger.info('Getting image data done.')

        self._size = self._labels.size
        self._variables = ('x', 'y')
        if rng is None:
            rng = numpy.random.RandomState(313)
        self.rng = rng
        self.reset()

    def reset(self):
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = numpy.arange(self._size)
        super(MnistDataSource, self).reset()

    @property
    def images(self):
        """Get copy of whole data with a shape of (N, 1, H, W)."""
        return self._images.copy()

    @property
    def labels(self):
        """Get copy of whole label with a shape of (N, 1)."""
        return self._labels.copy()


@contextmanager
def data_iterator_mnist(batch_size,
                        train=True,
                        rng=None,
                        shuffle=True,
                        with_memory_cache=False,
                        with_file_cache=False,
                        output_dir=None):
    '''
    Provide DataIterator with :py:class:`MnistDataSource`
    with_memory_cache, with_parallel and with_file_cache option's default value is all False,
    because :py:class:`MnistDataSource` is able to store all data into memory.

    For example,

    .. code-block:: python

        with data_iterator_mnist(True, batch_size) as di:
            for data in di:
                SOME CODE TO USE data.

    '''
    with MnistDataSource(train=train, shuffle=shuffle, rng=rng, output_dir=output_dir) as ds, \
        data_iterator(ds,
                      batch_size,
                      rng=None,
                      with_memory_cache=with_memory_cache,
                      with_file_cache=with_file_cache) as di:
        yield di


def data_iterator_to_csv(csv_path, csv_file_name, data_path, data_iterator):
    index = 0
    csv_data = []
    with data_iterator as data:
        line = ['x:image', 'y:label']
        csv_data.append(line)
        pbar = tqdm.tqdm(total=data.size, unit='images')
        initial_epoch = data.epoch
        while data.epoch == initial_epoch:
            d = data.next()
            for i in range(len(d[0])):
                label = d[1][i][0]
                file_name = data_path + \
                    '/{}'.format(label) + '/{}.png'.format(index)
                full_path = os.path.join(
                    csv_path, file_name.replace('/', os.path.sep))
                directory = os.path.dirname(full_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                imwrite(full_path, d[0][i].reshape(28, 28))
                csv_data.append([file_name, label])
                index += 1
                pbar.update(1)
        pbar.close()
    with open(os.path.join(csv_path, csv_file_name), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(csv_data)
    return csv_data


def create_onehot_dataset(source_csv, file_name, num_class):
    csv_data = []
    for i, line in enumerate(source_csv):
        if i == 0:
            new_line = [line[0]]
            for i2 in range(num_class):
                new_line.append('y__{}:{}'.format(i2, i2))
            csv_data.append(new_line)
        else:
            label = line[1]
            onehot = numpy.zeros(num_class)
            onehot[label] = 1
            new_line = [line[0]]
            new_line.extend(onehot)
            csv_data.append(new_line)
    with open(file_name, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(csv_data)


def create_data_csv(seed):
    path = os.path.abspath(os.path.dirname(__file__))
    base_dir = os.path.join(path, 'mnist')
    ensure_dir(base_dir)

    # Create original training set
    logger.log(99, 'Downloading MNIST training set images...')
    output_dir = os.path.join(path, 'download')
    train_di = data_iterator_mnist(
        60000, True, None, False, output_dir=output_dir)
    logger.log(99, 'Creating "mnist_training.csv"... ')
    train_csv = data_iterator_to_csv(
        base_dir, 'mnist_training.csv', 'training', train_di)

    # Create original test set
    logger.log(99, 'Downloading MNIST test set images...')
    validation_di = data_iterator_mnist(
        10000, False, None, False, output_dir=output_dir)
    logger.log(99, 'Creating "mnist_test.csv"... ')
    test_csv = data_iterator_to_csv(
        base_dir, 'mnist_test.csv', 'validation', validation_di)

    # Create one-hot training set
    logger.log(99, 'Creating "mnist_training_onehot.csv"... ')
    create_onehot_dataset(train_csv, os.path.join(
        base_dir, "mnist_training_onehot.csv"), 10)

    # Create one-hot test set
    logger.log(99, 'Creating "mnist_test_onehot.csv"... ')
    create_onehot_dataset(test_csv, os.path.join(
        base_dir, "mnist_test_onehot.csv"), 10)

    # Create 100 data training set for semi-supervised learning
    logger.log(99, 'Creating "mnist_training_100.csv"... ')
    labels = numpy.zeros(10)
    csv_data = []
    for i, line in enumerate(train_csv):
        if i == 0:
            csv_data.append(line)
        else:
            label = line[1]
            if labels[label] < 10:
                csv_data.append(line)
                labels[label] += 1
    with open(os.path.join(base_dir, "mnist_training_100.csv"), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(csv_data)

    # Create unlabeled training set for semi-supervised learning
    logger.log(99, 'Creating "mnist_training_unlabeled.csv"... ')
    labels = numpy.zeros(10)
    csv_data = []
    for i, line in enumerate(train_csv):
        if i == 0:
            csv_data.append(['xu:image'])
        else:
            csv_data.append([line[0]])
    with open(os.path.join(base_dir, "mnist_training_unlabeled.csv"), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(csv_data)

    # Create small training set
    logger.log(99, 'Creating "small_mnist_4or9_training.csv"... ')
    labels = numpy.zeros(2)
    csv_data = []
    for i, line in enumerate(train_csv):
        if i == 0:
            csv_data.append(['x:image', 'y:9'])
        else:
            label = line[1]
            if label == 4 or label == 9:
                if label == 4:
                    label = 0
                else:
                    label = 1
                if labels[label] < 750:
                    csv_data.append([line[0], label])
                    labels[label] += 1
    with open(os.path.join(base_dir, "small_mnist_4or9_training.csv"), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(csv_data)

    # Create small test set
    logger.log(99, 'Creating "small_mnist_4or9_test.csv"... ')
    labels = numpy.zeros(2)
    csv_data = []
    for i, line in enumerate(test_csv):
        if i == 0:
            csv_data.append(['x:image', 'y:9'])
        else:
            label = line[1]
            if label == 4 or label == 9:
                if label == 4:
                    label = 0
                else:
                    label = 1
                if labels[label] < 250:
                    csv_data.append([line[0], label])
                    labels[label] += 1
    with open(os.path.join(base_dir, "small_mnist_4or9_test.csv"), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(csv_data)

    # Create small test set with initial memory
    logger.log(99, 'Creating "small_mnist_4or9_test_w_initmemory.csv"... ')
    memory_size = 256
    for i in range(len(csv_data)):
        if i == 0:
            for i2 in range(memory_size):
                csv_data[0].append('c__{}'.format(i2))
        else:
            csv_data[i].extend(numpy.zeros(memory_size))
    with open(os.path.join(base_dir, "small_mnist_4or9_test_w_initmemory.csv"), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(csv_data)

    logger.log(99, 'Dataset creation completed successfully.')

    # split based on seed
    # Create original training set
    train_csv, val_csv = split_data_into_train_val(
        train_csv, val_size=10000, seed=seed)
    save_list_to_csv(train_csv, base_dir, 'mnist_training' +
                     '_' + str(seed) + '.csv')
    save_list_to_csv(val_csv, base_dir, 'mnist_validation' +
                     '_' + str(seed) + '.csv')


def main(args):
    create_data_csv(args.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='csv data', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '-s', '--seed', help='seed num', default=0, type=int, required=True)
    args = parser.parse_args()
    main(args)
