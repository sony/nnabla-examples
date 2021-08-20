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
Provide data iterator for CIFAR10 examples.
'''
from contextlib import contextmanager
import argparse
import numpy as np
import tarfile
import os
import tqdm
from imageio import imwrite
from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.data_source_loader import download
from .utils import get_filename_to_download, save_list_to_csv, split_data_into_train_val, ensure_dir


class Cifar10DataSource(DataSource):
    '''
    Get data directly from cifar10 dataset from Internet(yann.lecun.com).
    '''

    def _get_data(self, position):
        image = self._images[self._indexes[position]]
        label = self._labels[self._indexes[position]]
        return (image, label)

    def __init__(self, train=True, shuffle=False, rng=None, output_dir=None):
        super(Cifar10DataSource, self).__init__(shuffle=shuffle)

        self._train = train
        data_uri = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        logger.info('Getting labeled data from {}.'.format(data_uri))
        output_file = get_filename_to_download(output_dir, data_uri)
        r = download(data_uri, output_file=output_file)  # file object returned
        with tarfile.open(fileobj=r, mode="r:gz") as fpin:
            # Training data
            if train:
                images = []
                labels = []
                for member in fpin.getmembers():
                    if "data_batch" not in member.name:
                        continue
                    fp = fpin.extractfile(member)
                    data = np.load(fp, allow_pickle=True, encoding="bytes")
                    images.append(data[b"data"])
                    labels.append(data[b"labels"])
                self._size = 50000
                self._images = np.concatenate(
                    images).reshape(self._size, 3, 32, 32)
                self._labels = np.concatenate(labels).reshape(-1, 1)
            # Validation data
            else:
                for member in fpin.getmembers():
                    if "test_batch" not in member.name:
                        continue
                    fp = fpin.extractfile(member)
                    data = np.load(fp, allow_pickle=True, encoding="bytes")
                    images = data[b"data"]
                    labels = data[b"labels"]
                self._size = 10000
                self._images = images.reshape(self._size, 3, 32, 32)
                self._labels = np.array(labels).reshape(-1, 1)
        r.close()
        logger.info('Getting labeled data from {}.'.format(data_uri))

        self._size = self._labels.size
        self._variables = ('x', 'y')
        if rng is None:
            rng = np.random.RandomState(313)
        self.rng = rng
        self.reset()

    def reset(self):
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super(Cifar10DataSource, self).reset()

    @property
    def images(self):
        """Get copy of whole data with a shape of (N, 1, H, W)."""
        return self._images.copy()

    @property
    def labels(self):
        """Get copy of whole label with a shape of (N, 1)."""
        return self._labels.copy()


@contextmanager
def data_iterator_cifar10(batch_size,
                          train=True,
                          rng=None,
                          shuffle=True,
                          with_memory_cache=False,
                          with_file_cache=False,
                          output_dir=None):
    '''
    Provide DataIterator with :py:class:`Cifar10DataSource`
    with_memory_cache, with_parallel and with_file_cache option's default value is all False,
    because :py:class:`Cifar10DataSource` is able to store all data into memory.

    For example,

    .. code-block:: python

        with data_iterator_cifar10(True, batch_size) as di:
            for data in di:
                SOME CODE TO USE data.

    '''
    with Cifar10DataSource(train=train, shuffle=shuffle, rng=rng, output_dir=output_dir) as ds, \
        data_iterator(ds,
                      batch_size,
                      rng=rng,
                      with_memory_cache=with_memory_cache,
                      with_file_cache=with_file_cache) as di:
        yield di


def data_iterator_to_csv(csv_path, data_path, data_iterator, seed=0):
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
                imwrite(full_path, d[0][i].reshape(
                    3, 32, 32).transpose(1, 2, 0))
                csv_data.append([file_name, label])
                index += 1
                pbar.update(1)
        pbar.close()
    return csv_data


def create_data_csv(seed):
    path = os.path.abspath(os.path.dirname(__file__))
    base_dir = os.path.join(path, 'cifar10')
    ensure_dir(base_dir)
    # Create original training set
    logger.log(99, 'Downloading CIFAR-10 dataset...')
    output_dir = os.path.join(path, 'download')
    train_di = data_iterator_cifar10(50000, output_dir=output_dir)
    logger.log(99, 'Creating "cifar10_training.csv"... ')
    train_csv = data_iterator_to_csv(base_dir, 'training', train_di)
    train_csv, val_csv = split_data_into_train_val(
        train_csv, val_size=10000, seed=seed)

    save_list_to_csv(train_csv, base_dir,
                     'cifar10_training' + '_' + str(seed) + '.csv')
    save_list_to_csv(val_csv, base_dir, 'cifar10_validation' +
                     '_' + str(seed) + '.csv')
    # Create original test set
    validation_di = data_iterator_cifar10(
        10000, False, None, False, output_dir=output_dir)
    logger.log(99, 'Creating "cifar10_test.csv"... ')
    test_csv = data_iterator_to_csv(base_dir, 'validation', validation_di)
    save_list_to_csv(test_csv, base_dir, 'cifar10_test.csv')

    logger.log(99, 'Dataset creation completed successfully.')


def main(args):
    create_data_csv(args.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='csv data', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '-s', '--seed', help='seed num', default=0, type=int, required=True)
    args = parser.parse_args()
    main(args)
