'''
* 2nd challenge (to be updated!):
    - Max video label length: 18
    - The number of files (train): 3,888,919
    - The number of files (valid): 1,112,356
    
* 3rd challenge:
    - Max video label length: 14
    - Max video-segment label length: 4
    - Max segment label length: 15
    - The number of files (valid): 47,087 -> 38,786
    - The number of files (test): 44,753 -> 44,739
'''

import os
import numpy as np
import glob
import shutil
import torch

# data_types = ['train', 'valid', 'test']
# challenge = '2nd_challenge' / '3rd_challenge'

data_types = ['train']
challenge = '2nd_challenge'

data_dir = '/run/media/hoosiki/WareHouse1/mtb/datasets/VU/pytorch_datasets/'

max_vid_label_len = 0
max_vid_seg_label_len = 0
max_seg_label_len = 0
for data_type in data_types:
    frame_dir = data_dir + '{}/{}/'.format(challenge, data_type)
    bad_frame_dir = '/run/media/hoosiki/WareHouse2/mtb/datasets/bad_datasets/{}/'.format(challenge, data_type)
    os.makedirs(bad_frame_dir, exist_ok=True)
    file_paths = glob.glob(frame_dir + '*.pt')

    print('The number of files:', len(file_paths))
    for i, file_path in enumerate (file_paths):
        if i % 10000 == 0:
            print(data_type, i)
        try :
            data = torch.load(file_path)
            frame_len = len(data['frame_rgb'])
            vid_label_len = len(data['video_labels'])
            if (challenge == '2nd_challenge'):
                assert(data_type == 'train' or data_type == 'valid')
                if (frame_len <= 10 or vid_label_len == 0):
                    shutil.move(file_path, bad_frame_dir)
                else:
                    if max_vid_label_len < vid_label_len:
                        max_vid_label_len = vid_label_len
            else:
                vid_seg_label_len = len(data['video_segment_labels'])
                seg_label_len = len(data['segment_labels'])
                assert(data_type == 'valid' or data_type == 'test')
                if (data_type == 'valid'):
                    max_seg_start_times = 0 if vid_seg_label_len == 0 else max(data['segment_times'] * 5)
                    if (frame_len <= 10 or vid_seg_label_len == 0 or frame_len < max_seg_start_times):
                        shutil.move(file_path, bad_frame_dir)
                    else:
                        if max_vid_label_len < vid_label_len:
                            max_vid_label_len = vid_label_len
                        if max_vid_seg_label_len < vid_seg_label_len:
                            max_vid_seg_label_len = vid_seg_label_len
                        if max_seg_label_len < seg_label_len:
                            max_seg_label_len = seg_label_len
                else:
                    if (frame_len <= 10):
                        shutil.move(file_path, bad_frame_dir)
        except:
            shutil.move(file_path, bad_frame_dir)

print("Max video label length:", max_vid_label_len)
print("Max video-segment label length:", max_vid_seg_label_len)
print("Max segment label length:", max_seg_label_len)
print("Done!")
