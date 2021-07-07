from gluoncv.data.transforms import video
from gluoncv.data import VideoClsCustom
def video_transform(root, video_path, train=False, test_mode=False):
    if train == True:
        transform_train = video.VideoGroupTrainTransform(size=(224, 224), scale_ratios=[1.0, 0.8], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_dataset = VideoClsCustom(root=root,
                                    setting=video_path,
                                    train=True,
                                    test_mode=False,
                                    #num_segments=3,
                                    new_length=8,
                                    transform=transform_train,
                                    #video_loader=True,
                                    #use_decord=True
                                    )
        print('Load %d training samples.' % len(train_dataset))

    elif train == False and test_mode == True:
        transform_train = video.VideoGroupValTransform(size=(224, 224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_dataset = VideoClsCustom(root=root,
                                    setting=video_path,
                                    train=False,
                                    test_mode=True,
                                    #num_segments=3,
                                    new_length=8,
                                    transform=transform_train,
                                    #video_loader=True,
                                    #use_decord=True
                                    )
        print('Load %d testing samples.' % len(train_dataset))
    return train_dataset

