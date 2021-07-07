from PIL import Image, ImageDraw, ImageFont
import os
from natsort import natsorted
from itertools import chain
from moviepy.editor import *
import shutil
import pandas as pd


def visualization(project):
    file_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.abspath(os.path.join(file_path, '..', '..', project))
    result_path = os.path.join(file_path, 'result')
    concat_video = os.path.join(file_path, 'concat_video')
    ttf_path = os.path.join(file_path, 'preinstall', 'Roboto')
    labelled_image_for_concat = os.path.join(
        file_path, 'labelled_image_for_concat')
    if os.path.exists(concat_video):
        shutil.rmtree(concat_video)
    os.mkdir(concat_video)
    if os.path.exists(labelled_image_for_concat):
        shutil.rmtree(labelled_image_for_concat)
    os.mkdir(labelled_image_for_concat)

    category = pd.read_csv(os.path.join(result_path, 'predicted_result.csv'))[
        'predicted category']

    img_folder = os.path.join(file_path, 'test_frame')
    dir_list = os.listdir(img_folder)
    dir_list = natsorted(dir_list)
    dir_list_path = [os.path.join(img_folder, d) for d in dir_list]
    for i, img_fold in enumerate(dir_list_path):
        img_fold_dir_list = os.listdir(img_fold)
        img_fold_dir_list = natsorted(img_fold_dir_list)
        img_fold_dir_path_list = [os.path.join(
            img_fold, img) for img in img_fold_dir_list]
        for j, single_img in enumerate(img_fold_dir_path_list):
            # writing text on image
            img = Image.open(single_img)
            draw = ImageDraw.Draw(img)
            # create font object with the font file and specify
            # desired size

            font = ImageFont.truetype(os.path.join(
                ttf_path, 'Roboto-Bold.ttf'), size=100)
            # starting position of the message

            (x, y) = (50, 50)
            message = str(category[i])
            color = 'rgb(255, 255, 255)'  # black color
            # draw the message on the background
            draw.text((x, y), message, fill=color, font=font)
            img.save(os.path.join(labelled_image_for_concat,
                                  dir_list[i] + '_' + img_fold_dir_list[j]))
            print('---save img_', i)

    # image_list = os.listdir(labelled_image_for_concat)
    # image_list = natsorted(image_list)
    # image_list_path = [os.path.join(labelled_image_for_concat, img) for img in image_list]

    # #concat clips
    # clips = [ImageClip(m).set_duration(2) for m in image_list_path]
    # concat_clip = concatenate_videoclips(clips, method="compose")
    # concat_clip.write_videofile(os.path.join(concat_video, "labelled_video.mp4"), fps=5)
