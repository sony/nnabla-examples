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

import pickle
import numpy as np
import nnabla as nn
from nnabla.ext_utils import get_extension_context
import args
import classifier as clf
import data_loader as dl
from utils import utils


def get_data_settings(opt):
    """
    Get the data settings
    Args:
        opt : variables that containing values for all of your options
    Returns:
        variables which you need to test & validate
    """
    data_params = {
        "train_beg": opt['train_beg'],
        "valid_beg": opt['valid_beg'],
        "test_beg": opt['test_beg'],
    }
    data_setting = {
        'path': opt['base_img_path'],
        'protected_attribute': opt['protected_attribute'],
        'attribute': opt['attribute'],
        'data_params': data_params,
        'batch_size': opt['batch_size']
    }
    opt['data_setting'] = data_setting
    return opt


if __name__ == "__main__":

    opt = args.get_args()
    opt = get_data_settings(opt)
    attr_list = utils.get_all_attr()
    ctx = get_extension_context(
        opt['context'], device_id=opt['device_id'], type_config=opt['type_config'])
    nn.set_default_context(ctx)
    batch_size = opt['data_setting']['batch_size']
    test = dl.actual_celeba_dataset(opt['data_setting'], batch_size,
                                    augment=False, split='test', shuffle=False)
    AC = clf.attribute_classifier(model_load_path="{}/{}/best/best_acc.h5".format(
        opt['model_save_path'], attr_list[opt['attribute']]))
    val_results = pickle.load(open(r'{}/{}/best/val_results.pkl'.format(
        opt['model_save_path'], attr_list[opt['attribute']]), 'rb'))
    cal_thresh = val_results['cal_thresh']
    f1_thresh = val_results['f1_thresh']
    test_targets, test_scores = AC.get_scores(test)
    test_pred = np.where(test_scores > cal_thresh, 1, 0)
    ap = utils.get_average_precision(test_targets[:, 0], test_scores)
    deo = utils.get_difference_equality_opportunity(test_targets[:, 1],
                                                    test_targets[:, 0], test_pred)
    ba = utils.get_bias_amplification(test_targets[:, 1],
                                      test_targets[:, 0], test_pred)
    kl = utils.get_kl_divergence(test_targets[:, 1],
                                 test_targets[:, 0], test_scores)

    test_results = {
        'AP': ap,
        'DEO': deo,
        'BA': ba,
        'KL': kl,
        'f1_thresh': f1_thresh,
        'cal_thresh': cal_thresh,
        'opt': opt
    }
    print('Test results: ')
    print('AP : {:.1f} +- {:.1f}', 100 * ap)
    print('DEO : {:.1f} +- {:.1f}', 100 * deo)
    print('BA : {:.1f} +- {:.1f}', 100 * ba)
    print('KL : {:.1f} +- {:.1f}', kl)

    with open(r'{}/{}/best/test_scores.pkl'.format(
            opt['model_save_path'], attr_list[opt['attribute']]), 'wb+') as handle:
        pickle.dump(test_scores, handle)

    with open(r'{}/{}/best/test_targets.pkl'.format(
            opt['model_save_path'], attr_list[opt['attribute']]), 'wb+') as handle:
        pickle.dump(test_targets, handle)

    with open(r'{}/{}/best/test_results.pkl'.format(
            opt['model_save_path'], attr_list[opt['attribute']]), 'wb+') as handle:
        pickle.dump(test_results, handle)
