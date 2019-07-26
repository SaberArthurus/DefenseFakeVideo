import pandas as pd
import numpy as np
import cv2

features_key = ['AU01_r', 'AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', \
    'pose_Rx', 'pose_Rz', 'MOUTHh', 'MOUTHv'] 
# 'AU45_r' eye blink is useless, pitch (Rx), yaw (Ry), and roll (Rz).
center_coordinate = [i for i in range(18, 37)] + [49, 55]
head_coordinate = [i for i in range(0, 37)] + [49, 55]


def extract_features_from_openface(csv_file, frame_begin, frame_end, img_size):
    print(img_size)
    h, w ,_ = img_size
    cammeraMatrix = np.array([[w, 0, w*.5], [0, h, h*.5], [0, 0, 1]])
    image_coord = []
    world_coord = []
    # img_coord
    for idx in range(68):
        featx = np.array(csv_file.iloc[frame_begin:frame_end][' ' + 'x_{}'.format(idx)])
        featy = np.array(csv_file.iloc[frame_begin:frame_end][' ' + 'y_{}'.format(idx)])
        
        image_coord.append(np.stack([featx, featy], axis=0))

        featX = np.array(csv_file.iloc[frame_begin:frame_end][' ' + 'X_{}'.format(idx)])
        featY = np.array(csv_file.iloc[frame_begin:frame_end][' ' + 'Y_{}'.format(idx)])
        featZ = np.array(csv_file.iloc[frame_begin:frame_end][' ' + 'Z_{}'.format(idx)])
        world_coord.append(np.stack([featX, featY, featZ], axis=0))

    image_coord = np.stack(image_coord, axis=1).reshape(2, -1).transpose()
    image_coord = list(image_coord)
    world_coord = np.stack(world_coord, axis=1).reshape(3, -1).transpose()
    world_coord = list(world_coord)
    
    dist_coef = np.zeros(4)
    cv2.calibrateCamera(world_coord, image_coord, (img_size[:2])[::-1], cammeraMatrix, dist_coef)
    print(world_coord)

if __name__ == '__main__':
    filename = 'test.csv'
    csv_data = pd.read_csv(filename)
    image = cv2.imread('content/000000.jpg')
    extract_features_from_openface(csv_data, 1, 3, image.shape)
