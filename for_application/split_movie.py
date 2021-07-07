from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
import shutil


def split_movie(project):
    file_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.abspath(os.path.join(file_path, '..', '..', project))
    video_path = os.path.abspath(os.path.join(file_path, r'original_video'))
    label_path = os.path.abspath(os.path.join(file_path, r'video_label'))
    split_path = os.path.abspath(os.path.join(file_path, r'split_video'))
    if os.path.exists(split_path):
        shutil.rmtree(split_path)
    os.mkdir(split_path)

    required_video_file = os.path.join(video_path, "catch_wire_3.mp4")

    with open(os.path.join(label_path, "video_label.txt")) as f:
        times = f.readlines()

    times = [x.strip() for x in times]

    for time in times:
        starttime = float(time.split("-")[0])
        endtime = float(time.split("-")[1])
        ffmpeg_extract_subclip(required_video_file, starttime, endtime, targetname=os.path.join(
            split_path, str(times.index(time)+1)+".mp4"))
