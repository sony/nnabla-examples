import re
from pathlib import Path
import glob

import numpy as np
import os
import fnmatch

from PIL import Image

from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.image_utils import imread, imresize


from . import _get_sliced_data_source

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)): 
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

##################################################
# data loader / iterator
##################################################

def load_function(path, image_shape, sketch=False):
    if(sketch):
        image = imread(path, channel_first=True, grayscale=True, num_channels=1)
    else:
        image = imread(path, channel_first=True)

    if image.shape[1:] != image_shape:
        # imresize takes (width, height) as shape.
        resize_shape = (image_shape[1], image_shape[0])
        image = imresize(image, resize_shape, channel_first=True)

    # normalize
    image = (image - 127.5) / 127.5  # -> [-1, 1]

    return image

class tcvcIterator(DataSource):
    DATASET_NAME = 'GUNDAM'
    DATASET_ROOT_PATH = './datasets/' + DATASET_NAME


    def __init__(self, dataset_path=None, image_shape=(256, 256), shuffle=True, rng=None, flip=True, dataset_mode="train", is_train=True):
        super(tcvcIterator, self).__init__(shuffle=shuffle, rng=rng)
        
        print("dataset_mode: ", dataset_mode)
        if dataset_path is None:
            dataset_path = tcvcIterator.DATASET_ROOT_PATH + '/' + dataset_mode
        else: 
            dataset_path =dataset_path + '/' + dataset_mode

        self._image_dir = os.path.join(dataset_path, 'color')
        self._sketch_dir = os.path.join(dataset_path, 'sketch')    
        
        if is_train:
            self._image_path = make_dataset(self._image_dir)
            self._sketch_path = make_dataset(self._sketch_dir)
        else:
            self._image_path = sorted(make_dataset(self._image_dir))
            self._sketch_path = sorted(make_dataset(self._sketch_dir))
            
        self._image_shape = image_shape
        self._size = len(self._image_path)
        self._variables = ("image", "image_prev", "sketch")
        self.flip = flip
        self.is_train = is_train
        self.reset()

    def reset(self):
        self._idxs = self._rng.permutation(
            self._size) if self.shuffle else np.arange(self._size)

        super(tcvcIterator, self).reset()

    def __iter__(self):
        self.reset()
        return self

    def _get_data(self, position):

        idx = self._idxs[position]
        image_path = self._image_path[idx]
        sketch_path = self._sketch_path[idx]

        image = load_function(image_path, self._image_shape) 
        sketch = load_function(sketch_path, self._image_shape, True) 
        image_prev = self.get_prev(image_path, is_train=self.is_train)

        if self.flip:
            if np.random.rand() > 0.5:
                image = image[..., ::-1]
                image_prev = image_prev[..., ::-1]
                sketch = sketch[..., ::-1]

        return image, image_prev, sketch

    def get_prev(self, file_path, is_train=True):
        frame_number, _ = self.get_frame_number(file_path)
        previous_frame_file_path = self.get_previous_frame_file_path(file_path)

        if frame_number == 0 or not previous_frame_file_path.exists():
            initial_prev = Image.new("RGB", [256, 256])
            initial_prev = np.array(initial_prev).transpose(2,0,1)
            initial_prev = (initial_prev - 127.5) / 127.5  
            return initial_prev
        else:
            # if statement <0.5 return blank image
            rnd = np.random.uniform(0, 1)
            if rnd <= 0.5 or (not is_train): 
                prev = load_function(previous_frame_file_path, self._image_shape)
            else:
                prev = Image.new("RGB", [256, 256])
                prev = np.array(prev).transpose(2,0,1)
                prev = (prev - 127.5) / 127.5  

            return prev

    @staticmethod
    def get_frame_number(file_path):
        filename = Path(file_path).name
        m = re.search(r"\D*(\d{1,7})\.(jpg|jpeg|png)$", filename)
        if m:
            padded_frame_number_as_string = m.group(1)
            frame_number = 0
            stripped_number_as_string = m.group(1).lstrip("0")  # remove leading zeroes
            if len(stripped_number_as_string) > 0:
                frame_number = int(stripped_number_as_string)
            return frame_number, padded_frame_number_as_string
        raise Exception('Could not find a frame number in "{}"'.format(file_path))

    @staticmethod
    def get_previous_frame_file_path(file_path):
        file_path = Path(file_path)
        frame_number, padded_frame_number_as_string = tcvcIterator.get_frame_number(file_path.name)
        num_digits = len(padded_frame_number_as_string)
        format_string = "{{:0{}d}}".format(num_digits)
        padded_previous_frame_number = format_string.format(frame_number - 1)
        previous_frame_file_path = file_path.with_name(
            file_path.name.replace(
                padded_frame_number_as_string, padded_previous_frame_number
            )
        )
        return previous_frame_file_path



def create_data_iterator(batch_size, dataset_path, image_shape, comm=None, shuffle=True, rng=None,
                         with_memory_cache=False, with_parallel=False, with_file_cache=False, flip=True, 
                         dataset_mode="train", is_train=True):

    ds = tcvcIterator(dataset_path, image_shape, shuffle=shuffle, rng=rng, flip=flip, 
                        dataset_mode=dataset_mode, is_train=is_train)

    ds = _get_sliced_data_source(ds, comm, shuffle=shuffle)

    return data_iterator(ds,
                         batch_size,
                         with_memory_cache,
                         with_parallel,
                         with_file_cache)
