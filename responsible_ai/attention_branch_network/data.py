import argparse
import numpy as np
import tarfile
import os
import tqdm
import csv
from imageio import imwrite
from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.data_source_loader import download


class Cifar10DataSource(DataSource):
    '''
    Get data directly from cifar10 dataset from Internet(yann.lecun.com).
    '''

    def _get_data(self, position):
        image = self._images[self._indexes[position]]
        label = self._labels[self._indexes[position]]
        return (image, label)

    def __init__(self, train=True, shuffle=False, rng=None):
        super(Cifar10DataSource, self).__init__(shuffle=shuffle)

        self._train = train
        data_uri = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        logger.info('Getting labeled data from {}.'.format(data_uri))
        r = download(data_uri)  # file object returned
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


def data_iterator_cifar10(batch_size,
                          train=True,
                          rng=None,
                          shuffle=True,
                          with_memory_cache=False,
                          with_file_cache=False):
    '''
    Provide DataIterator with :py:class:`Cifar10DataSource`
    with_memory_cache, with_parallel and with_file_cache option's default value is all False,
    because :py:class:`Cifar10DataSource` is able to store all data into memory.

    For example,
    '''

    ds = Cifar10DataSource(train=train, shuffle=shuffle, rng=rng)
    di = data_iterator(ds, batch_size, rng=rng,
                       with_memory_cache=with_memory_cache, with_file_cache=with_file_cache)
    return di


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
                imwrite(full_path, d[0][i].reshape(
                    3, 32, 32).transpose(1, 2, 0))
                csv_data.append([file_name, label])
                index += 1
                pbar.update(1)
        pbar.close()
    with open(os.path.join(csv_path, csv_file_name), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(csv_data)
    return csv_data


def func(args):
    path = args.output_dir

    # Create original training set
    logger.log(99, 'Downloading CIFAR-10 dataset...')
    train_di = data_iterator_cifar10(50000, True, None, False)
    logger.log(99, 'Creating "cifar10_training.csv"... ')
    data_iterator_to_csv(
        path, 'cifar10_training.csv', './training', train_di)

    # Create original test set
    validation_di = data_iterator_cifar10(10000, False, None, False)
    logger.log(99, 'Creating "cifar10_test.csv"... ')
    data_iterator_to_csv(
        path, 'cifar10_test.csv', './validation', validation_di)

    logger.log(99, 'Dataset creation completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='CIFAR10\n\n' +
        'Download CIFAR-10 dataset from dl.sony.com (original file is from https://www.cs.toronto.edu/~kriz/cifar.html).\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-o',
        '--output-dir',
        help='path to write NNC dataset CSV format (dir) default=CIFAR10',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
