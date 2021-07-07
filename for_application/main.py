from split_movie import split_movie
from video_dur_txt import create_video_duration_txt
from fine_tune_for_application import fine_tune
from video2frame import video_to_frame


from auto_labelling_for_application import get_label_df
from prediction_for_application import prediction
from visualization import visualization

project = 'test2'
nclass = 2
video_name = 'catch_wire_3.mp4'

create_video_duration_txt(project, video_name, mode=1)
split_movie(project)
video_to_frame(project, application=True)

get_label_df(project)
fine_tune(project, nclass, epochs=3)
prediction(project, 2, nclass)  # change it for the best params
visualization(project)
