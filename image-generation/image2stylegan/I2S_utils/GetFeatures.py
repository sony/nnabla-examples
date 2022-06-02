# Copyright 2022 Sony Group Corporation.
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
import nnabla as nn
from nnabla.utils.nnp_graph import NnpNetworkPass
from nnabla.models.imagenet.base import ImageNetBase


def get_feature_keys(_pre_trained_model):
    key_list = []
    if _pre_trained_model == VGG16_custom:
        key_list = [
            'VGG16/ReLU',               # shape=(64,256,256)
            'VGG16/ReLU_2',             # shape=(64,256,256)
            'VGG16/ReLU_6',             # shape=(256,64,64)
            'VGG16/ReLU_9'              # shape=(512,32,32)
        ]
    elif _pre_trained_model == MobileNet_custom:
        key_list = [
            'ReLU',                     # shape=(32,128,128)
            'ReLU_3',                   # shape=(64,128,128)
            'ReLU_7',                   # shape=(128,64,64)
            'ReLU_11',                  # shape=(256,32,32)
        ]
    return key_list


class MobileNet_custom(ImageNetBase):
    """
    MobileNet architecture.
    The following is a list of string that can be specified to ``use_up_to`` option in ``__call__`` method;
    * ``'classifier'`` (default): The output of the final affine layer for classification.
    * ``'pool'``: The output of the final global average pooling.
    * ``'lastconv'``: The input of the final global average pooling without ReLU activation.
    * ``'lastconv+relu'``: Network up to ``'lastconv'`` followed by ReLU activation.
    References:
        * `Howard et al., MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.
          <https://arxiv.org/abs/1704.04861>`_
    """

    _KEY_VARIABLE = {
        'classifier': 'Affine',
        'pool': 'AveragePooling',
        'lastconv': 'BatchNormalization_19',
        'lastconv+relu': 'ReLU_19',
        }

    def __init__(self):
        self._load_nnp('MobileNet.nnp', 'MobileNet/MobileNet.nnp')

    def _input_shape(self):
        return (3, 224, 224)

    def __call__(self, input_var=None, use_from=None, use_up_to='classifier', training=False, force_global_pooling=False, check_global_pooling=True, returns_net=False, verbose=0):

        input_var = self.get_input_var(input_var)

        callback = NnpNetworkPass(verbose)
        callback.drop_function('ImageAugmentationX')
        callback.set_variable('ImageAugmentationX', input_var)
        self.configure_global_average_pooling(
            callback, force_global_pooling, check_global_pooling, 'AveragePooling')
        callback.set_batch_normalization_batch_stat_all(training)
        self.use_up_to(use_up_to, callback)
        if not training:
            callback.fix_parameters()
        batch_size = input_var.shape[0]
        net = self.nnp.get_network(
            'Training', batch_size=batch_size, callback=callback)
        if returns_net:
            return net
        return list(net.outputs.values())[0]


class VGG_custom(ImageNetBase):

    """
    VGG architectures for 11, 13, 16 layers.
    Args:
        num_layers (int): Number of layers chosen from 11, 13, 16.
    The following is a list of string that can be specified to ``use_up_to`` option in ``__call__`` method;
    * ``'classifier'`` (default): The output of the final affine layer for classification.
    * ``'pool'``: The output of the final global average pooling.
    * ``'lastconv'``: The input of the final global average pooling without ReLU activation.
    * ``'lastconv+relu'``: Network up to ``'lastconv'`` followed by ReLU activation.
    * ``'lastfeature'``: Network up to one layer before ``'classifier'``, but without activation.
    References:
        * `Simonyan and Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition.
          <https://arxiv.org/pdf/1409.1556>`_
    """

    def __init__(self, num_layers=11):

        # Check validity of num_layers
        set_num_layers = set((11, 13, 16))
        assert num_layers in set_num_layers, "num_layers must be chosen from {}".format(
            set_num_layers)
        self.num_layers = num_layers

        # Load nnp
        self._load_nnp('VGG-{}.nnp'.format(num_layers),
                       'VGG-{0}/VGG-{0}.nnp'.format(num_layers))

        self._KEY_VARIABLE = {
            'classifier': 'VGG{}/Affine_3'.format(num_layers),
            'pool': 'VGG{}/MaxPooling_5'.format(num_layers),
            'lastconv': 'VGG16/Convolution_13' if num_layers == 16 else 'VGG{}/Convolution_12'.format(num_layers),
            'lastconv+relu': 'VGG16/ReLU_13' if num_layers == 16 else 'VGG{}/ReLU_12'.format(num_layers),
            'lastfeature': 'VGG{}/Affine_2'.format(num_layers),
            }

    def _input_shape(self):
        return (3, 224, 224)

    def __call__(self, input_var=None, use_from=None, use_up_to='classifier', training=False,
                 force_global_pooling=False, check_global_pooling=True, returns_net=False, verbose=0):

        assert use_from is None, 'This should not be set because it is for forward compatibility.'
        input_var = self.get_input_var(input_var)

        callback = NnpNetworkPass(verbose)
        callback.drop_function('ImageAugmentationX')
        callback.set_variable('ImageAugmentationX', input_var)
        callback.set_batch_normalization_batch_stat_all(training)
        self.use_up_to(use_up_to, callback)
        if not training:
            callback.drop_function(
                'VGG{}/Dropout_1'.format(self.num_layers))
            callback.drop_function(
                'VGG{}/Dropout_2'.format(self.num_layers))
            callback.fix_parameters()
        batch_size = input_var.shape[0]
        net = self.nnp.get_network(
            'Training', batch_size=batch_size, callback=callback)
        if returns_net:
            return net
        return list(net.outputs.values())[0]


class VGG16_custom(VGG_custom):
    def __init__(self):
        super(VGG16_custom, self).__init__(16)
