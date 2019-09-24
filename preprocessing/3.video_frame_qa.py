'''
* 2nd challenge (to be updated!):
    - Max video label length: 18
    - The number of files (train): 3,888,919
    - The number of files (valid): 1,112,356
    
* 3rd challenge:
    - Max video label length: 4
    - Max segment label length: 15
    - The number of files (valid): 47,087 -> 38,788
    - The number of files (test): 44,753 -> 44,739
'''

import os
import numpy as np
import glob
import shutil

# data_types = ['train', 'valid', 'test']
# challenge = '2nd_challenge' / '3rd_challenge'

data_types = ['valid']
challenge = '3rd_challenge'

data_dir = '/run/media/hoosiki/WareHouse2/mtb/datasets/VU/pytorch_datasets/'

max_vid_label_len = 0
max_seg_label_len = 0
for data_type in data_types:
    frame_dir = data_dir + '{}/{}/'.format(challenge, data_type)
    bad_frame_dir = data_dir + '{}/bad_datasets/{}/'.format(challenge, data_type)
    os.makedirs(bad_frame_dir, exist_ok=True)
    file_paths = glob.glob(frame_dir + '*.npy')

    print('The number of files:', len(file_paths))
    for i, file_path in enumerate (file_paths):
        if i % 10000 == 0:
            print(data_type, i)
        data = np.load(file_path, allow_pickle=True).item()
        frame_rgb_len = len(data['frame_rgb'])
        frame_audio_len = len(data['frame_audio'])
        vid_label_len = len(data['video_labels'])
        if (frame_rgb_len != frame_audio_len):
            print('(ERROR) - check if frame length is correct or not!')
            break
        if (challenge == '2nd_challenge'):
            assert(data_type == 'train' or data_type == 'valid')
            if (frame_rgb_len <= 30 or vid_label_len == 0 or frame_rgb_len >= 302):
                shutil.move(file_path, bad_frame_dir)
            else:
                if max_vid_label_len < vid_label_len:
                    max_vid_label_len = vid_label_len
        else:
            seg_label_len = len(data['segment_labels'])
            assert(data_type == 'valid' or data_type == 'test')
            if (data_type == 'valid'):
                max_seg_start_times = 0 if vid_label_len == 0 else max(data['segment_times'] * 5)
                if (frame_rgb_len <= 10 or vid_label_len == 0 or frame_rgb_len < max_seg_start_times):
                    shutil.move(file_path, bad_frame_dir)
                else:
                    if max_vid_label_len < vid_label_len:
                        max_vid_label_len = vid_label_len
                    if max_seg_label_len < seg_label_len:
                        max_seg_label_len = seg_label_len
            else:
                if (frame_rgb_len <= 10):
                    shutil.move(file_path, bad_frame_dir)

print("Max video label length:", max_vid_label_len)
print("Max segment label length:", max_seg_label_len)
print("Done!")
