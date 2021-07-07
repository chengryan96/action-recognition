import numpy as np
import pandas as pd
import os
import re
import csv
import shutil
import yaml
from natsort import natsorted
from sklearn.model_selection import train_test_split
from create_folder import create_folder


def get_label_df(project):
    file_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.abspath(os.path.join(file_path, '..', '..', project))
    # clear existing folder or create new
    label_folder = os.path.join(file_path, 'label_txt')
    create_folder(label_folder)
    video_path = os.path.join(file_path, r'frame')
    yaml_path = os.path.join(file_path, r'yaml', r'category_pattern.yaml')
    dir_list = os.listdir(video_path)
    # find the number of the frame
    dir_len_list = [len(os.listdir(os.path.join(
        video_path, dir_list[i])))-1 for i in range(len(dir_list))]
    # we can define the number of frame too
    #dir_len_list = [17 for i in range(len(dir_list))]

    with open(yaml_path) as file:
        category_pattern = yaml.load(file, Loader=yaml.FullLoader)

    try:
        dir_list.remove(r'.DS_Store')  # mac only
    except:
        pass
    full_list = []
    for category in category_pattern:
        for i in range(len(dir_list)):
            if bool(re.match(category, dir_list[i])) == True:
                full_list.append(
                    [video_path + r'/' + dir_list[i], dir_len_list[i], category_pattern[category]])
    full_df = pd.DataFrame(full_list)

    # testing files are not useful in this case
    train, val = train_test_split(
        full_df, test_size=0.2, random_state=4001, shuffle=True)

    train.to_csv(label_folder + r'/train.txt', sep=' ', index=False,
                 header=False, quoting=csv.QUOTE_NONE, quotechar="",  escapechar="\\")

    val.to_csv(label_folder + r'/val.txt', sep=' ', index=False,
               header=False, quoting=csv.QUOTE_NONE, quotechar="",  escapechar="\\")

    ################################################################################################
    # for prediction
    # no label

    file_path = os.path.abspath(os.path.join(file_path, '..', project))
    video_path = os.path.abspath(os.path.join(file_path, r'test_frame'))
    dir_list = os.listdir(video_path)
    dir_list = natsorted(dir_list)
    dir_len_list = [len(os.listdir(os.path.join(
        video_path, dir_list[i])))-1 for i in range(len(dir_list))]
    #dir_len_list = [17 for i in range(len(dir_list))]
    category_pattern = {
        'None': '999'
    }

    # dir_list.sort()
    try:
        dir_list.remove(r'.DS_Store')  # mac only
    except:
        pass
    full_list = []
    for category in category_pattern:
        for i in range(len(dir_list)):
            full_list.append([video_path + r'/' + dir_list[i],
                              dir_len_list[i], category_pattern[category]])
    full_df = pd.DataFrame(full_list)
    test = full_df
    test.to_csv(label_folder + r'/test.txt', sep=' ', index=False,
                header=False, quoting=csv.QUOTE_NONE, quotechar="",  escapechar="\\")


project = 'catch_wire_1_app'
get_label_df(project)
