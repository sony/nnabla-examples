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

if __name__ == "__main__":

    opt = args.get_args()
    attr_list = utils.get_all_attr()
    ctx = get_extension_context(
        opt['context'], device_id=opt['device_id'], type_config=opt['type_config'])
    nn.set_default_context(ctx)
    generated_image_range = (0, 175000)  # total generated images 175000
    loader = dl.fake_dataset_no_label("{}/AllGenImages/".format(opt["fake_data_dir"]),
                                      generated_image_range,
                                      batch_size=opt['batch_size'],
                                      shuffle=False)

    AC = clf.attribute_classifier(model_load_path="{}/{}/best/best_acc.h5".format(
        opt['model_save_path'], attr_list[opt['attribute']]))
    _, scores = AC.get_scores(loader)
    threshold = pickle.load(open("{}/{}/best/val_results.pkl".format(
        opt['model_save_path'], attr_list[opt['attribute']]), 'rb'))['f1_thresh']
    scores = np.where(scores > threshold, 1.0, 0.0)
    with open("{}/all_{}_scores.pkl".format(opt['fake_data_dir'],
                                            attr_list[opt['attribute']]), 'wb+') as handle:
        pickle.dump(scores, handle)
