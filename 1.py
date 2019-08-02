import mmcv
video = mmcv.VideoReader('test.avi')
video.cvt2frames('content')
