"""\
MLP example.
"""

import argparse
import sys

from pyeddl.api import (
    Input, Activation, Dense, Model, sgd, CS_CPU, build, T_load, div, fit,
    evaluate, CS_GPU, BatchNormalization, L1, L2, L1L2
)
from pyeddl.utils import download_mnist


def main(args):
    download_mnist()

    epochs = args.epochs
    batch_size = args.batch_size
    num_classes = 10

    in_ = Input([784])
    layer = in_
    layer = BatchNormalization(
        Activation(Dense(layer, 1024, True, L1(0.01)), "relu")
    )
    layer = BatchNormalization(
        Activation(Dense(layer, 1024, True, L2(0.01)), "relu")
    )
    layer = BatchNormalization(
        Activation(Dense(layer, 1024, True, L1L2(0.01, 0.01)), "relu")
    )
    out = Activation(Dense(layer, num_classes), "softmax")
    net = Model([in_], [out])

    build(
        net,
        sgd(0.01, 0.9),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        CS_GPU([1]) if args.gpu else CS_CPU(4)
    )

    print(net.summary())

    x_train = T_load("trX.bin")
    y_train = T_load("trY.bin")
    x_test = T_load("tsX.bin")
    y_test = T_load("tsY.bin")

    div(x_train, 255.0)
    div(x_test, 255.0)

    fit(net, [x_train], [y_train], batch_size, epochs)
    evaluate(net, [x_test], [y_test])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=1000)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
