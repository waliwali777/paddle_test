# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np

from paddle import fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear

from model import Model, CrossEntropy, Input, set_device
from metrics import Accuracy
from tsm import *

NUM_CLASSES = 10


def make_optimizer(num_samples, parameter_list=None):
    step = int(num_samples / FLAGS.batch_size / FLAGS.num_devices)
    boundaries = [e * step for e in [40, 60]]
    values = [FLAGS.lr * (0.1 ** i) for i in range(len(boundaries) + 1)]

    learning_rate = fluid.layers.piecewise_decay(
        boundaries=boundaries,
        values=values)
    optimizer = fluid.optimizer.Momentum(
        learning_rate=learning_rate,
        regularization=fluid.regularizer.L2Decay(1e-4),
        momentum=0.9,
        parameter_list=parameter_list)

    return optimizer


def main():
    device = set_device(FLAGS.device)
    fluid.enable_dygraph(device) if FLAGS.dynamic else None

    train_transform = Compose([GroupScale(),
                               GroupMultiScaleCrop(),
                               GroupRandomCrop(),
                               GroupRandomFlip(),
                               NormalizeImage()])
    train_dataset = KineticsDataset(
            filelist=os.path.join(FLAGS.data, 'train_10.list'),
            pickle_dir=os.path.join(FLAGS.data, 'train_10'),
            transform=train_transform)
    val_transform = Compose([GroupScale(),
                             GroupCenterCrop(),
                             NormalizeImage()])
    val_dataset = KineticsDataset(
            filelist=os.path.join(FLAGS.data, 'val_10.list'),
            pickle_dir=os.path.join(FLAGS.data, 'val_10'),
            mode='val',
            transform=val_transform)

    pretrained = FLAGS.eval_only and FLAGS.weights is None
    model = tsm_resnet50(num_classes=NUM_CLASSES, pretrained=pretrained)
    optim = make_optimizer(len(train_dataset), model.parameters())

    inputs = [Input([None, 8, 3, 224, 224], 'float32', name='image')]
    labels = [Input([None, 1], 'int64', name='label')]

    model.prepare(
        optim,
        CrossEntropy(),
        metrics=Accuracy(topk=(1, 5)),
        inputs=inputs,
        labels=labels,
        device=FLAGS.device)

    if FLAGS.eval_only:
        if FLAGS.weights is not None:
            model.load(FLAGS.weights)

        model.evaluate(
            val_dataset,
            batch_size=FLAGS.batch_size,
            num_workers=FLAGS.num_workers)
        return

    if FLAGS.resume is not None:
        model.load(FLAGS.resume)

    model.fit(train_data=train_dataset,
              eval_data=val_dataset,
              epochs=FLAGS.epoch,
              batch_size=FLAGS.batch_size,
              save_dir='tsm_checkpoint',
              num_workers=4,
              drop_last=True,
              shuffle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CNN training on TSM")
    parser.add_argument('data', metavar='DIR', help='path to kineteics dataset')
    parser.add_argument(
        "--device", type=str, default='gpu', help="device to use, gpu or cpu")
    parser.add_argument(
        "-d", "--dynamic", action='store_true', help="enable dygraph mode")
    parser.add_argument(
        "--eval_only", action='store_true', help="run evaluation only")
    parser.add_argument(
        "-e", "--epoch", default=70, type=int, help="number of epoch")
    parser.add_argument(
        "-j", "--num_workers", default=4, type=int, help="read worker number")
    parser.add_argument(
        '--lr',
        '--learning-rate',
        default=1e-2,
        type=float,
        metavar='LR',
        help='initial learning rate')
    parser.add_argument(
        "-b", "--batch_size", default=16, type=int, help="batch size")
    parser.add_argument(
        "-n", "--num_devices", default=1, type=int, help="number of devices")
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="checkpoint path to resume")
    parser.add_argument(
        "-w",
        "--weights",
        default=None,
        type=str,
        help="weights path for evaluation")
    FLAGS = parser.parse_args()
    main()
