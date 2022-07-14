import os
import pickle
import numpy as np
from sklearn import metrics
from nnabla.ext_utils import get_extension_context
import nnabla as nn
import args
import classifier as clf
import data_loader as di
from utils import utils

if __name__ == '__main__':

    opt = args.get_args()
    ctx = get_extension_context(
        opt['context'], device_id=opt['device_id'], type_config=opt['type_config'])
    nn.set_default_context(ctx)
    batch_size = opt['batch_size']
    model_to_load = os.path.join(
        opt['model_save_path'], opt['attribute'], 'best_acc.h5')
    val_result_file = os.path.join(
        opt['model_save_path'], opt['attribute'], 'val_results.pkl')
    if not (os.path.exists(model_to_load) and os.path.exists(val_result_file)):
        print(f'Provided model path : {model_to_load}')
        print(f'Provided val result file : {val_result_file}')
        raise ('Provided model save path is not proper')
    rng = np.random.RandomState(1)
    # load test data
    test_loader = di.data_iterator_celeba(
        opt['celeba_image_test_dir'], opt['attr_path'], batch_size,
        target_attribute=opt['attribute'], protected_attribute=opt['protected_attribute'],
        rng=rng)
    nn.clear_parameters()
    attribute_classifier_model = clf.AttributeClassifier(
        model_load_path=model_to_load)
    cal_thresh = pickle.load(open(val_result_file, 'rb'))['cal_thresh']
    test_targets, test_scores = attribute_classifier_model.get_scores(
        test_loader)
    test_pred = np.where(test_scores > cal_thresh, 1, 0)
    accuracy = metrics.accuracy_score(test_targets[:, 0], test_pred)
    dpd, eod, aaod = utils.get_fairness(
        test_targets[:, 0], test_targets[:, 1] == 1, test_pred)
    utils.plot_fairness_multi(dpd, eod, aaod, accuracy, bar_x_axis="original")
    print(f"Before applying ROC(Test set) : \n\
                                    Demographic Parity Difference (DPD) : {round(dpd * 100, 2)} \n\
                                    Equal Opportunity Difference (EOD) : {round(eod * 100, 2)}\n\
                                    Absolute Average Odd Difference (AAOD): {round(aaod * 100, 2)}\n\
                                    Accuracy : {round(accuracy * 100, 2)}")
