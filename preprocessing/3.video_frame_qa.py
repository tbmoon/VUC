# 2nd challenge
# The number of files (train): 3,888,919

import os
import numpy as np
import glob
import shutil

# data_types = ['train', 'valid', 'test']
# challenge = '2nd_challenge' / '3rd_challenge'

data_types = ['train']
challenge = '2nd_challenge'

data_dir = '/run/media/hoosiki/WareHouse2/mtb/datasets/VU/pytorch_datasets/'

for data_type in data_types:
    frame_dir = data_dir + '{}/{}/'.format(challenge, data_type)
    bad_frame_dir = data_dir + '{}/bad_datasets/{}/'.format(challenge, data_type)
    os.makedirs(bad_frame_dir, exist_ok=True)
    file_paths = glob.glob(frame_dir + '*.npy')

    print('The number of files:', len(file_paths))
    for i, file_path in enumerate (file_paths):
        if i % 1000 == 0:
            print(data_type, i)
        data = np.load(file_path, allow_pickle=True).item()
        frame_rgb_len = len(data['frame_rgb'])
        frame_audio_len = len(data['frame_audio'])
        video_label_len = len(data['video_labels'])
        if (frame_rgb_len != frame_audio_len):
            print('(ERROR) - check if frame length is correct or not!')
            break
        if (challenge == '2nd_challenge'):
            assert(data_type == 'train' or data_type == 'valid')
            if (frame_rgb_len <= 10 or video_label_len == 0 or frame_rgb_len >= 302):
                #print(file_path)
                shutil.move(file_path, bad_frame_dir)
        else:
            assert(data_type == 'valid' or data_type == 'test')
            if (data_type == 'valid'):
                max_segment_start_times = max(data['segment_start_times'])
                if (frame_rgb_len <= 10 or video_label_len == 0 or frame_rgb_len < max_segment_start_times):
                    print(file_path)
                    shutil.move(file_path, bad_frame_dir)
            else:
                if (frame_rgb_len <= 10):
                    print(file_path)
                    shutil.move(file_path, bad_frame_dir)
print("Done!")
