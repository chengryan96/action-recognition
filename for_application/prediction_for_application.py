from __future__ import print_function

import mxnet as mx
import mxnet.ndarray as nd
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model
import numpy as np
import os
from video_config import video_transform
from gluoncv.utils import split_and_load
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv
# config

project = 'catch_wire_1_app'

#prediction(project, 1, 2)


def prediction(project, best_epoch, nclass, num_gpus=1,     num_workers=4):
    file_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.abspath(os.path.join(file_path, '..', '..', project))
    params_folder = os.path.join(file_path, 'params')
    result_folder = os.path.join(file_path, 'result')
    plot_folder = os.path.join(file_path, 'plot')

    ctx = [mx.gpu(i) for i in range(num_gpus)]

    def load_params(best_epoch, model, nclass):
        file_name = 'net.params_' + str(best_epoch)
        net = get_model(name=model, nclass=nclass)  # change nclass
        net.collect_params().reset_ctx(ctx)  # use gpu
        # load params
        net.load_parameters(os.path.join(
            file_path, params_folder, file_name), ctx=ctx)
        print('the params are loaded')
        return net

    net = load_params(best_epoch, 'i3d_resnet50_v1_custom', 2)

    root = os.path.join(file_path, 'video')
    video_setting = os.path.join(file_path, 'label_txt/test.txt')
    test_dataset = video_transform(root, video_setting, test_mode=True)
    batch_size = 1

    # set 1 batch and turn shuffle off meaning loading all data according to the order of test.txt
    test = gluon.data.DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)

    y_pred = []
    predict_result_detail = []
    for i, batch in enumerate(test):
        # Extract data and label
        data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        for _, X in enumerate(data):
            X = X.reshape((-1,) + X.shape[2:])
            pred = net(X)
            classes = [i for i in range(0, nclass)]  # need to fix
            topK = 1
            ind = nd.topk(pred, k=topK)[0].astype('int')
            y_pred.append(ind.asnumpy()[0])
            print('The input video clip is classified to be')
            for i in range(topK):
                print('\t[%s], with probability %.3f.' %
                      (classes[ind[i].asscalar()], nd.softmax(pred)[0][ind[i]].asscalar()))
        predict_result_detail.append(
            [ind.asnumpy()[0], nd.softmax(pred)[0][ind[i]].asscalar()])

    predict_result_detail_df = pd.DataFrame(predict_result_detail, columns=[
                                            'predicted category', 'probability'])
    predict_result_detail_df.to_csv(result_folder + r'/predicted_result.csv',
                                    index=False, quoting=csv.QUOTE_NONE, quotechar="",  escapechar="\\")
