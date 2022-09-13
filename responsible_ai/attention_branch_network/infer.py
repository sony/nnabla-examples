import functools
from tqdm import tqdm
import nnabla as nn
from nnabla.ext_utils import get_extension_context
from responsible_ai.attention_branch_network.data import data_iterator_cifar10
from responsible_ai.attention_branch_network.resnet110 import resnet110
from responsible_ai.attention_branch_network.abn import abn


def infer(model="abn", parameter="./params_30.h5", device_id=0, type_config='float'):
    extension_module = "cudnn"
    ctx = get_extension_context(
        extension_module, device_id=device_id, type_config=type_config
    )
    nn.set_default_context(ctx)

    mch = 0
    path = parameter
    nn.clear_parameters()

    input_data = data_iterator_cifar10(
        1, train=False, shuffle=False)

    x = nn.Variable((1, 3, 32, 32))
    x.data.zero()

    t = nn.Variable((1, 1))

    if model == "resnet110":
        prediction = functools.partial(resnet110)
        nn.load_parameters(path)
        y = prediction(x=x, test=True)
    elif model == "abn":
        prediction = functools.partial(abn)
        nn.load_parameters(path)
        _, y = prediction(x=x, test=True)

    nn.get_parameters()

    for _ in tqdm(range(input_data.size)):
        x.d, t.d = input_data.next()
        y.forward()

        if t.d == y.d.argmax(axis=1):
            mch += 1

    print("Accuracy:{}".format(mch / input_data.size))
