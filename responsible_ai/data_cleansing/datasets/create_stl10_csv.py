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

import os
import numpy as np
from contextlib import contextmanager
from tqdm import tqdm
from imageio import imwrite
import csv
from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.data_source_loader import download
import tarfile
import argparse
from .utils import split_data_into_train_val, save_list_to_csv, get_filename_to_download, ensure_dir


class STL10DataSource(DataSource):
    '''
    Directly get data from STL10
    implemented refering to this example
    https://github.com/sony/nnabla/blob/master/tutorial/cifar10_classification.ipynb
    '''

    def __init__(self, train=True, shuffle=False, rng=None, output_dir=None):
        super(STL10DataSource, self).__init__(shuffle=shuffle, rng=rng)
        self._train = train
        data_uri = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
        logger.info('Getting labbeled data from {}'.format(data_uri))
        default_seed = 313
        output_file = get_filename_to_download(output_dir, data_uri)
        r = download(data_uri, output_file=output_file)  # file object returned
        print(r.name)
        binary_dir = os.path.join(output_dir, "stl10_binary")
        with tarfile.open(fileobj=r, mode="r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=output_dir)

        for member in os.listdir(binary_dir):
            if train:
                if 'train_' not in member:
                    continue
                print(member)
                self.load_image_and_labels(os.path.join(binary_dir, member))

            # Validation data
            else:
                print(member)
                if 'test_' not in member:
                    continue
                self.load_image_and_labels(os.path.join(binary_dir, member))
        r.close
        logger.info('Getting labeled data from {}'.format(data_uri))
        self._size = self._labels.size
        self._variables = ('x', 'y')
        if rng is None:
            rng = np.random.RandomState(default_seed)
        self.rng = rng
        self.reset()

    def load_image_and_labels(self, path_to_file):
        if '_X' in path_to_file:
            print("images")
            self._images = self.load_images(path_to_file)

        elif '_y' in path_to_file:
            print("labels")
            self._labels = self.load_labels(path_to_file)

    def load_images(self, image_binary):
        with open(image_binary, "rb") as fp:
            images = np.fromfile(fp, dtype=np.uint8)
            images = images.reshape(-1, 3, 96, 96)
            return images

    def load_labels(self, label_binary):
        with open(label_binary, "rb") as fp:
            labels = np.fromfile(fp, dtype=np.uint8)
            labels = labels.reshape(-1, 1)
            # 1-10 -> 0-9
            labels = labels - 1
            return labels

    def _get_data(self, position):
        image = self._images[self._indices[position]]
        label = self._labels[self._indices[position]]
        return (image, label)

    def reset(self):
        if self._shuffle:
            self._indices = self.rng.permutation(self._size)
        else:
            self._indices = np.arange(self._size)
        super(STL10DataSource, self).reset()

    @property
    def images(self):
        '''Get copy of whole data with a shape of (N, 1, H, W).'''
        return self._images.copy()

    @property
    def labels(self):
        '''Get copy of whole label with a shape of (N, 1).'''
        return self._labels.copy()


@contextmanager
def data_iterator_stl10(batch_size,
                        train=True,
                        rng=None,
                        shuffle=True,
                        with_memory_cache=False,
                        with_file_cache=False,
                        output_dir=None):
    '''
    Provide DataIterator with :py:class:`STL10DataSource`
    with_memory_cache and with_file_cache option's default value is all False,
    because :py:class:`STL10DataSource` is able to store all data into memory.
    '''
    """
    _data_iterator = data_iterator(
        STL10DataSource(train=train, shuffle=shuffle, rng=rng),
        batch_size,
        rng,
        with_memory_cache,
        with_file_cache
    )
    return _data_iterator
    """

    with STL10DataSource(train=train, shuffle=shuffle, rng=rng, output_dir=output_dir) as ds, \
        data_iterator(ds,
                      batch_size,
                      rng=rng,
                      with_memory_cache=with_memory_cache,
                      with_file_cache=with_file_cache) as di:
        yield di


def data_iterator_to_csv(csv_path, csv_file_name, data_path, data_iterator):
    index = 0
    csv_data = []
    with data_iterator as data:
        line = ['x:image', 'y:label']
        csv_data.append(line)
        pbar = tqdm(total=data.size, unit='images')
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
                    3, 96, 96).transpose(1, 2, 0))
                csv_data.append([file_name, label])
                index += 1
                pbar.update(1)
        pbar.close()
    with open(os.path.join(csv_path, csv_file_name), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(csv_data)
    return csv_data


def create_data_csv(seed):
    path = os.path.abspath(os.path.dirname(__file__))
    base_dir = os.path.join(path, 'stl10')
    ensure_dir(base_dir)
    # Create original training set
    logger.log(99, 'Downloading STL10 dataset...')
    output_dir = os.path.join(path, 'download')
    train_di = data_iterator_stl10(
        5000, True, None, False, output_dir=output_dir)
    logger.log(99, 'Creating "stl10_training.csv"... ')
    train_csv = data_iterator_to_csv(
        base_dir, 'stl10_training.csv', 'training', train_di)
    train_csv, val_csv = split_data_into_train_val(
        train_csv, val_size=1000, seed=seed)
    save_list_to_csv(train_csv, base_dir, 'stl10_training' +
                     '_' + str(seed) + '.csv')
    save_list_to_csv(val_csv, base_dir, 'stl10_validation' +
                     '_' + str(seed) + '.csv')

    # Validation
    validation_di = data_iterator_stl10(
        8000, False, None, False, output_dir=output_dir)
    logger.log(99, 'Creating "stl10_test.csv"... ')
    _ = data_iterator_to_csv(
        base_dir, 'stl10_test.csv', 'validation', validation_di)
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
