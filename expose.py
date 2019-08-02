import pandas as pd
import numpy as np
import cv2

features_key = ['AU01_r', 'AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', \
    'pose_Rx', 'pose_Rz', 'MOUTHh', 'MOUTHv'] 
# 'AU45_r' eye blink is useless, pitch (Rx), yaw (Ry), and roll (Rz).
center_coordinate = [i for i in range(18, 37)] + [49, 55]
head_coordinate = [i for i in range(0, 37)] + [49, 55]


def extract_features_from_openface(csv_file, frame_begin, frame_end, im, mode='global'):
    # print(img_size)
    size = im.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
    image_coord = []
    world_coord = []
    if mode == 'global':
        select_idx = [i for i in range(1, 37)] + [49, 55]
    else:
        select_idx = [i for i in range(18, 37)] + [49, 55]

    for idx in select_idx:
        featx = np.array(csv_file.iloc[frame_begin:frame_end][' ' + 'x_{}'.format(idx)])
        featy = np.array(csv_file.iloc[frame_begin:frame_end][' ' + 'y_{}'.format(idx)])
        
        image_coord.append(np.stack([featx, featy], axis=0))

        featX = np.array(csv_file.iloc[frame_begin:frame_end][' ' + 'X_{}'.format(idx)])
        featY = np.array(csv_file.iloc[frame_begin:frame_end][' ' + 'Y_{}'.format(idx)])
        featZ = np.array(csv_file.iloc[frame_begin:frame_end][' ' + 'Z_{}'.format(idx)])
        world_coord.append(np.stack([featX, featY, featZ], axis=0))


    image_coord = np.stack(image_coord, axis=1).reshape(1, 1, 2, -1).transpose((0, 3, 1, 2)).astype(np.float32).squeeze(0).squeeze(1)
    world_coord = np.stack(world_coord, axis=1).reshape(1, 3, -1).transpose((0, 2, 1)).astype(np.float32).squeeze(0)
    dist_coeffs = np.zeros((1, 4))
    retval, rvec, tvec = cv2.solvePnP(world_coord, image_coord, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1)]), rvec, tvec, camera_matrix, dist_coeffs)
    # print(nose_end_point2D)
    return nose_end_point2D[0][0]
    '''
    dvec = np.zeros((3, 3))
    cv2.Rodrigues(rvec, dvec)
    rvec = np.array(dvec)
    w = np.array([0, 0, 1]).T
    '''


def cos_similarity(a, b):
    return a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == '__main__':
    import mmcv
    demo_name = 'demo_fake'
    video = mmcv.VideoReader('{}.mp4'.format(demo_name))
    video.cvt2frames('content')

    filename = '{}.csv'.format(demo_name)
    csv_data = pd.read_csv(filename)
    for i in range(100):
        image = cv2.imread('content/%06d.jpg'%(i))
        va = extract_features_from_openface(csv_data, i+1, i+2, image, mode='global')
        vc = extract_features_from_openface(csv_data, i+1, i+2, image, mode='local')
        print(1-cos_similarity(va, vc))
    
    