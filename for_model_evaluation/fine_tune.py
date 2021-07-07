#import library
from __future__ import division
import pandas as pd
import argparse
import time
import logging
import os
import sys
import math
import csv
import numpy as np
import mxnet as mx
import gluoncv as gcv
from mxnet import gluon, nd, init, context
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.data.transforms import video
from gluoncv.data import VideoClsCustom
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRSequential, LRScheduler, split_and_load, TrainingHistory

from video_config import video_transform
import matplotlib.pyplot as plt
import shutil

from create_folder import create_folder


def fine_tune(project, nclass, epochs=100, num_gpus=1, per_device_batch_size=8, num_workers=4):
    # define the number of gpus

    # file path
    file_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.abspath(os.path.join(file_path, '..', '..', project))

    # this is to check if gpu is correctly install
    mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
    # to assign which device to run
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    # batch size per worker
    per_device_batch_size = 8
    # parallel computing
    num_workers = 4
    batch_size = per_device_batch_size * num_gpus
    root = os.path.join(file_path, 'frame')
    video_setting_train = os.path.join(file_path, 'label_txt/train.txt')
    train_dataset = video_transform(root, video_setting_train, train=True)
    train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, timeout=120, last_batch='discard', pin_memory=True)

    video_setting_val = os.path.join(file_path, 'label_txt/val.txt')
    val_dataset = video_transform(
        root, video_setting_val, train=False, test_mode=True)
    val_data = gluon.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, timeout=120, last_batch='discard', pin_memory=True)

    # for validation
    def test(ctx, val_data):

        metric = mx.metric.Accuracy()
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(
                batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(
                batch[1], ctx_list=ctx, batch_axis=0)
            output = []
            for _, X in enumerate(data):
                X = X.reshape((-1,) + X.shape[2:])
                pred = net(X)
                output.append(pred)
            metric.update(label, output)
        return metric.get()

    # custom newwork
    # number of category

    net = get_model(name='i3d_resnet50_v1_custom',
                    nclass=nclass)  # change nclass
    net.collect_params().reset_ctx(ctx)

    # create folder contains the result and each epoch
    result_folder = os.path.join(file_path, 'result')
    create_folder(result_folder)
    # create folder contains the parameters
    params_folder = os.path.join(file_path, 'params')
    create_folder(params_folder)

    # create folder contains plot
    plot_folder = os.path.join(file_path, 'plot')
    create_folder(plot_folder)

    ####################### Tune the parameters here ####################

    # Learning rate decay factor
    lr_decay = 0.1
    # Epochs where learning rate decays
    lr_decay_epoch = [5, 10, 15, 20, 25, 30, 35, 40,
                      45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

    # Stochastic gradient descent
    optimizer = 'sgd'
    # Set parameters
    optimizer_params = {'learning_rate': 0.001, 'wd': 0.0001, 'momentum': 0.9}

    # Define our trainer for net
    # for updating the params with the chosen optimization method
    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
    # loss function
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

    train_metric = mx.metric.Accuracy()
    train_history = TrainingHistory(['training-acc'])
    train_record = []  # to save the output into txt

    # training
    lr_decay_count = 0

    for epoch in range(epochs):
        tic = time.time()
        train_metric.reset()
        train_loss = 0
        loss_list = []
        # Learning rate decay
        if epoch == lr_decay_epoch[lr_decay_count]:
            trainer.set_learning_rate(trainer.learning_rate*lr_decay)
            lr_decay_count += 1
            #print('Learning rate decay')
        # Loop through each batch of training data
        for i, batch in enumerate(train_data):
            # Extract data and label
            data = split_and_load(
                batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            #print('Finished extract data')
            label = split_and_load(
                batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            #print('Finished extract label')
            # AutoGrad
            with ag.record():
                output = []
                for _, X in enumerate(data):
                    X = X.reshape((-1,) + X.shape[2:])
                    pred = net(X)
                    output.append(pred)
                loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]
                #print('finished AutoGrad')
            # Backpropagation
            for l in loss:
                l.backward()
                #print('finished Backpropagation')
            # Optimize
            trainer.step(batch_size)

            # Update metrics
            train_loss += sum([l.mean().asscalar() for l in loss])
            train_metric.update(label, output)
            #print('finished Update metrics')
            if i == 100:
                break
            #print(str(i) + r'iteration')
        name, acc = train_metric.get()
        # validation
        name, val_acc = test(ctx, val_data)
        # Update history and print metrics
        train_history.update([acc])
        print('[Epoch %d] train=%f loss=%f val_accuracy=%f time: %f' %
              (epoch, acc, train_loss / (i+1), val_acc, time.time()-tic))
        train_record.append(
            [epoch, acc, train_loss / (i+1), val_acc, time.time()-tic])
        loss_list.append(train_loss / (i+1))
        file_name = r'net.params_' + str(epoch)
        net.save_parameters(os.path.join(file_path, params_folder, file_name))

    # output the result into csv
    train_record_df = pd.DataFrame(train_record, columns=[
                                   'epoch', 'accuracy', 'loss', 'val-accuracy', 'time_taken'])
    train_record_df.to_csv(result_folder + r'/result_record.csv', index=False,
                           quoting=csv.QUOTE_NONE, quotechar="",  escapechar="\\")

    # We can plot the metric scores with:
    plt.style.use('ggplot')
    plt.figure()
    plt.title('train-accuracy')
    plt.xlabel('epoch')
    plt.xticks(range(len(list(train_record_df['epoch']))))
    plt.ylabel('validation-accuracy')
    plt.plot(list(train_record_df['epoch']), list(train_record_df['accuracy']))
    plt.savefig(os.path.join(file_path, plot_folder,
                             'validation-accuracy.jpg'))

    plt.figure()
    plt.title('validation-accuracy')
    plt.xlabel('epoch')
    plt.xticks(range(len(list(train_record_df['epoch']))))
    plt.ylabel('validation-accuracy')
    plt.plot(list(train_record_df['epoch']),
             list(train_record_df['val-accuracy']))
    plt.savefig(os.path.join(file_path, plot_folder, 'val-accuracy.jpg'))

    plt.figure()
    plt.title('loss')
    plt.xlabel('epoch')
    plt.xticks(range(len(list(train_record_df['epoch']))))
    plt.ylabel('loss')
    plt.plot(list(train_record_df['epoch']), list(train_record_df['loss']))
    plt.savefig(os.path.join(file_path, plot_folder, 'loss.jpg'))
