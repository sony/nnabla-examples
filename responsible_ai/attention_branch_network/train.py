import os
import functools
import math
import numpy as np
from tqdm import tqdm
import nnabla as nn
import nnabla.functions as F
import nnabla.solvers as S
import nnabla.utils.save as save
from nnabla.ext_utils import get_extension_context
from nnabla.utils.data_iterator import data_iterator_csv_dataset
from nnabla.utils.learning_rate_scheduler import StepScheduler
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
from responsible_ai.attention_branch_network.data import data_iterator_cifar10, data_iterator_to_csv
from responsible_ai.attention_branch_network.resnet110 import resnet110
from responsible_ai.attention_branch_network.abn import abn
from utils.neu.save_nnp import save_nnp


def categorical_error(pred, label):
    pred_label = pred.argmax(1)
    return (pred_label != label.flat).mean()


def loss_function(pred, label):
    loss = F.mean(F.softmax_cross_entropy(pred, label))
    return loss


def train(batch_size=128, device_id=0, type_config='float',
          model_save_path="./", train_epochs=30, model="abn"):
    bs_train = batch_size
    bs_valid = batch_size
    extension_module = "cudnn"
    ctx = get_extension_context(
        extension_module, device_id=device_id, type_config=type_config
    )
    nn.set_default_context(ctx)

    n_train_samples = 50000
    n_val_samples = 10000

    # Data Iterator
    train_di = data_iterator_cifar10(50000, True, None, False)
    train_csv = data_iterator_to_csv(
        './', 'cifar10_training.csv', './training', train_di)
    # Create original test set
    validation_di = data_iterator_cifar10(10000, False, None, False)
    test_csv = data_iterator_to_csv(
        './', 'cifar10_test.csv', './validation', validation_di)

    train_loader = data_iterator_csv_dataset(
        './cifar10_training.csv', batch_size=bs_train, shuffle=True, rng=np.random.RandomState(0), normalize=False, with_file_cache=False)
    val_loader = data_iterator_csv_dataset(
        './cifar10_test.csv', batch_size=bs_valid, shuffle=False, normalize=False, with_file_cache=False)

    if model == "resnet110":
        prediction = functools.partial(resnet110)

        # Create training graphs
        test = False
        image_train = nn.Variable((bs_train, 3, 32, 32))
        label_train = nn.Variable((bs_train, 1))
        pred_train = prediction(x=image_train, test=test)
        loss_train = loss_function(pred_train, label_train)

        # Create validation graph
        test = True
        image_valid = nn.Variable((bs_valid, 3, 32, 32))
        label_valid = nn.Variable((bs_valid, 1))
        pred_valid = prediction(x=image_valid, test=test)
        loss_val = loss_function(pred_valid, label_valid)

    elif model == "abn":
        prediction = functools.partial(abn)

        # Create training graphs
        test = False
        image_train = nn.Variable((bs_train, 3, 32, 32))
        label_train = nn.Variable((bs_train, 1))
        pred_att, pred_train = prediction(x=image_train, test=test)
        loss_att = loss_function(pred_att, label_train)
        loss_per = loss_function(pred_train, label_train)
        loss_train = loss_att + loss_per

        # Create validation graph
        test = True
        image_valid = nn.Variable((bs_valid, 3, 32, 32))
        label_valid = nn.Variable((bs_valid, 1))
        _, pred_valid = prediction(x=image_valid, test=test)
        loss_val = loss_function(pred_valid, label_valid)

    for param in nn.get_parameters().values():
        param.grad.zero()

    solver = S.Momentum(lr=0.1, momentum=0.9)
    solver.set_parameters(nn.get_parameters())

    # Learning rate scheduler
    first_step = 150
    second_step = 225
    iters_one_epoch = 390
    learning_rate_scheduler = StepScheduler(
        init_lr=0.1, gamma=0.1, iter_steps=[iters_one_epoch*first_step, iters_one_epoch*second_step])

    start_point = 0

    # Create monitor
    monitor = Monitor('tmp.monitor')
    monitor_loss = MonitorSeries("Training loss", monitor, interval=1)
    monitor_err = MonitorSeries("Training error", monitor, interval=1)
    monitor_time = MonitorTimeElapsed("Training time", monitor, interval=1)
    monitor_verr = MonitorSeries("Test error", monitor, interval=1)
    monitor_vloss = MonitorSeries("Test loss", monitor, interval=1)

    # save_nnp
    contents = save_nnp({"x": image_valid}, {"y": pred_valid}, bs_valid)
    save.save(
        os.path.join(model_save_path,
                     (model+"_epoch0_result.nnp")), contents
    )

    train_iter = math.ceil(n_train_samples / bs_train)
    val_iter = math.ceil(n_val_samples / bs_valid)

    min_loss = np.inf

    # Training-loop
    for i in tqdm(range(start_point, train_epochs)):
        # Forward/Zerograd/Backward
        print("## Training")
        e = 0.0
        loss = 0.0
        for k in range(train_iter):
            total_iter = train_iter * i + k
            lr = learning_rate_scheduler.get_learning_rate(total_iter)
            solver.set_learning_rate(lr)
            image, label = train_loader.next()
            image_train.d = image
            label_train.d = label
            loss_train.forward(clear_no_need_grad=True)
            solver.zero_grad()
            loss_train.backward(clear_buffer=True)
            solver.weight_decay(decay_rate=0.0001)
            solver.update()
            e += categorical_error(pred_train.d, label_train.d)
            loss += loss_train.data.data.copy() * bs_train
        e /= train_iter
        loss /= n_train_samples

        monitor_loss.add(i, loss)
        monitor_err.add(i, e)
        monitor_time.add(i)

        # Validation
        ve = 0.0
        vloss = 0.0

        print("## Validation")
        for j in range(val_iter):
            image, label = val_loader.next()
            image_valid.d = image
            label_valid.d = label
            loss_val.forward()
            vloss += loss_val.data.data.copy() * bs_valid
            ve += categorical_error(pred_valid.d, label)
        ve /= val_iter
        vloss /= n_val_samples
        monitor_verr.add(i, ve)
        monitor_vloss.add(i, vloss)

        if min_loss > vloss:
            nn.save_parameters(os.path.join(
                model_save_path, "{}_best_params.h5".format(model)))
            min_loss = vloss

        if i == 5 or i % 10 == 0:
            contents = save_nnp({"x": image_valid}, {
                                "y": pred_valid}, bs_valid)
            save.save(os.path.join(model_save_path,
                                   (model+"_result_{}.nnp".format(i))), contents)

    # save_nnp_lastepoch
    contents = save_nnp({"x": image_valid}, {"y": pred_valid}, bs_valid)
    save.save(os.path.join(model_save_path,
              (model+"_result_{}.nnp".format(train_epochs))), contents)
