import os
import shutil

import cv2
from natsort import natsorted


def video_to_frame(project, application=False):
    file_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.abspath(os.path.join(file_path, '..', '..', project))
    video_path = os.path.abspath(os.path.join(file_path, r'video'))
    testframe_folder = os.path.join(file_path, 'video', '..', 'test_frame')
    if os.path.exists(testframe_folder):
        shutil.rmtree(testframe_folder)
    os.mkdir(testframe_folder)

    # create folder
    frame_folder = os.path.join(file_path, 'video', '..', 'frame')
    if os.path.exists(frame_folder):
        shutil.rmtree(frame_folder)
    os.mkdir(frame_folder)

    dir_list = os.listdir(video_path)
    dir_list_name = [vid.replace('.mp4', '') for vid in dir_list]
    for vid in dir_list_name:
        label_folder = os.path.join(
            file_path, 'video', '..', 'frame', str(vid))
        if os.path.exists(label_folder):
            shutil.rmtree(label_folder)
        os.mkdir(label_folder)

    # convert labelled video to frame
    for i, vid in enumerate(dir_list):
        vidcap = cv2.VideoCapture(video_path + r'/' + str(vid))
        print(i, video_path + r'/' + str(vid))
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(os.path.join(video_path, '..', 'frame', str(
                dir_list_name[i])) + "/img_%05d.jpg" % count, image)     # save frame as JPEG file
            success, image = vidcap.read()
            print('Read a new frame: ', success)
            count += 1

    # convert test video to frame
    if application == True:
        video_path = os.path.abspath(os.path.join(file_path, r'split_video'))
        dir_list = os.listdir(video_path)
        dir_list = natsorted(dir_list)
        dir_list_name = [vid.replace('.mp4', '') for vid in dir_list]
        for vid in dir_list_name:
            label_folder = os.path.join(
                file_path, 'video', '..', 'test_frame', str(vid))
            if os.path.exists(label_folder):
                shutil.rmtree(label_folder)
            os.mkdir(label_folder)

        for i, vid in enumerate(dir_list):
            vidcap = cv2.VideoCapture(video_path + r'/' + str(vid))
            print(i, video_path + r'/' + str(vid))
            success, image = vidcap.read()
            count = 0
            while success:
                cv2.imwrite(os.path.join(video_path, '..', 'test_frame', str(
                    dir_list_name[i])) + "/img_%05d.jpg" % count, image)     # save frame as JPEG file
                success, image = vidcap.read()
                print('Read a new frame: ', success)
                count += 1
