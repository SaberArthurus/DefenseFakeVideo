import cv2
import numpy as np
import pandas as pd

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
    noise_center_idx = select_idx.index(33)

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
    
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rvec, tvec) = cv2.solvePnP(world_coord, image_coord, camera_matrix, dist_coeffs) # , flags=cv2.SOLVEPNP_ITERATIVE)

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rvec, tvec, camera_matrix, dist_coeffs)
    for p in image_coord:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
 
 
    p1 = ( int(image_coord[noise_center_idx][0]), int(image_coord[noise_center_idx][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    if mode == 'global':
        cv2.line(im, p1, p2, (255,0,0), 2) # 'blue
    else:
        cv2.line(im, p1, p2, (0,0,255), 2)
    
    cv2.imshow("Output", im)
    cv2.waitKey(0)


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
        # print(1-cos_similarity(va, vc))
    
    