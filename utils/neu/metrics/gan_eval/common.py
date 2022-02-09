from typing import List

from nnabla.utils.image_utils import imread
from nnabla.utils.data_iterator import data_iterator_simple


def get_data_iterator(path_list: List[str], batch_size: int, shuffle=False, stop_exhausted=False,
                      channel_first=True, num_channels=3):

    def load_func(i):
        path = path_list[i]

        # In the previous implementation, imageio is used to load image as below:
        # from imageio import imread
        # image = imread(path)
        # But, when we have diverse channel size (e.g. greyscale is also included) in dataset, this causes error in the following process.
        # If you think your metrics is warse than expected, using the previous one might help.

        image = imread(path, num_channels=num_channels,
                       channel_first=channel_first)
        return (image, )

    return data_iterator_simple(load_func=load_func,
                                num_examples=len(path_list),
                                batch_size=batch_size,
                                shuffle=shuffle,
                                stop_exhausted=stop_exhausted)
