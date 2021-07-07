from fine_tune import fine_tune
from video2frame import video_to_frame
from auto_labelling import get_label_df
from prediction import prediction


project = 'test1'
nclass = 2

video_to_frame(project, application=False)
get_label_df(project)
fine_tune(project, nclass, epochs=3)
prediction(project, 2, nclass)  # change it to use the best params
