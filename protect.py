import pandas as pd
import numpy as np

features_key = ['AU01_r', 'AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', \
    'pose_Rx', 'pose_Rz', 'MOUTHh', 'MOUTHv'] 
# 'AU45_r' eye blink is useless, pitch (Rx), yaw (Ry), and roll (Rz).


def extract_features_from_openface(csv_file, frame_begin, frame_end):
    '''
    frame >= 1
    '''
    feat_dict = {}
    for key in features_key:
        if 'MOUTH' not in key:
            feat_dict[key] = np.array(csv_file.iloc[frame_begin:frame_end][' ' + key])
    mouthx_list = []
    mouthy_list = []
    for i in range(48, 68): # the mouth world cordinate locates in from 48th to 68th point
        xi = np.array(csv_file.iloc[frame_begin:frame_end][' X_{}'.format(i)])
        yi = np.array(csv_file.iloc[frame_begin:frame_end][' Y_{}'.format(i)])
        mouthx_list.append(xi)
        mouthy_list.append(yi)
    mouthx = np.stack(mouthx_list)
    mouthy = np.stack(mouthy_list)
    
    feat_dict['MOUTHh'] = np.max(mouthx, axis=0) - np.min(mouthx, axis=0) 
    feat_dict['MOUTHv'] = np.max(mouthy, axis=0) - np.min(mouthy, axis=0) 
    return feat_dict



if __name__ == '__main__':
    filename = 'test.csv'
    csv_data = pd.read_csv(filename)
    feat_dict = extract_features_from_openface(csv_data, 1, 10)
    # print(feat_dict)
    for idx1, key1 in enumerate(features_key):
        for idx2, key2 in enumerate(features_key):
            if idx1 < idx2:
                print(key1, key2)
                print(np.corrcoef(feat_dict[key1], feat_dict[key2]))



