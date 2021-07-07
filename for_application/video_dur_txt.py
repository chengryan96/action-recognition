from moviepy.editor import VideoFileClip
import os
import numpy as np
import pandas as pd
import csv
import shutil


def create_video_duration_txt(project, video_name, mode=1):
    file_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.abspath(os.path.join(file_path, '..', '..', project))
    video_path = os.path.abspath(os.path.join(file_path, r'original_video'))
    label_folder = os.path.join(file_path, 'video_label')
    if os.path.exists(label_folder):
        shutil.rmtree(label_folder)
    os.mkdir(label_folder)

    clip = VideoFileClip(os.path.join(video_path, video_name))

    if mode == 1:
        # create a list per 0.5
        seq_np = np.arange(0, clip.duration, 0.5)
        seq_list = [str(seq_np[i]) + r'-' + str(seq_np[i+1])
                    for i in range(len(seq_np)-1)]
        seq_df = pd.DataFrame(seq_list)

        seq_df.to_csv(label_folder + r'/video_label.txt', sep=' ', index=False,
                      header=False, quoting=csv.QUOTE_NONE, quotechar="",  escapechar="\\")

    elif mode == 2:
        # second case
        # create a list per 0.5
        seq_np = np.arange(0, clip.duration, 0.25)
        seq_list = [str(seq_np[i]) + r'-' + str(seq_np[i+2])
                    for i in range(len(seq_np)-2)]
        seq_df = pd.DataFrame(seq_list)

        seq_df.to_csv(label_folder + r'/video_label.txt', sep=' ', index=False,
                      header=False, quoting=csv.QUOTE_NONE, quotechar="",  escapechar="\\")
