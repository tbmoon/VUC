import os
import numpy as np
import pandas as pd
import glob
import tensorflow as tf
'''
Purpose:
    1) tfrecord to be converted into npy format for pytorch running.
    2) label index is converted into index ranging from 0 to 1000.
'''
is_validate = True # True/False for validation/test sets, respectively.

base_dir = '/run/media/hoosiki/WareHouse3/mtb/datasets/VU/'
input_dir = base_dir + 'tfrecord_datasets/3rd_challenge/frame/'

if is_validate == True:
    datatype = 'validate'
else:
    datatype = 'test'
    
file_paths = glob.glob(input_dir + '{}*.tfrecord'.format(datatype))
out_dir   = base_dir + 'pickled_datasets/testing2/{}/'.format(datatype)
os.makedirs(out_dir, exist_ok=True)

#df_vocab = pd.read_csv(input_dir + 'vocabulary.csv')
#vocab_label2idx_dict = {0: 0}
#for i, label in enumerate(df_vocab['Index']):
#    vocab_label2idx_dict[label] = i+1

def parser(record):
    context_features = {
        'id': tf.io.FixedLenFeature([], tf.string),
        'labels': tf.io.VarLenFeature(tf.int64),
        'segment_start_times': tf.io.VarLenFeature(tf.int64),
        'segment_end_times': tf.io.VarLenFeature(tf.int64),
        'segment_labels': tf.io.VarLenFeature(tf.int64),
        'segment_scores': tf.io.VarLenFeature(tf.float32)        
    }
    sequence_features = {
        'rgb': tf.io.FixedLenSequenceFeature([], tf.string),
        'audio': tf.io.FixedLenSequenceFeature([], tf.string)
    }
    contexts, sequences = tf.io.parse_single_sequence_example(record,
                                                              context_features=context_features,
                                                              sequence_features=sequence_features)
    video_id = contexts['id']
    video_labels = contexts['labels']
    segment_start_times = contexts['segment_start_times']
    segment_end_times = contexts['segment_end_times']
    segment_labels = contexts['segment_labels']
    segment_scores = contexts['segment_scores']
    frame_rgb = tf.reshape(tf.decode_raw(sequences['rgb'], tf.uint8), [-1, 1024])
    frame_audio = tf.reshape(tf.decode_raw(sequences['audio'], tf.uint8), [-1, 128])
    return video_id, video_labels, segment_start_times, segment_end_times, segment_labels, segment_scores, frame_rgb, frame_audio

with tf.compat.v1.Session() as sess:
    for ifile in range(0, len(file_paths)):
        frame_lvl_record = input_dir + '{}%04d.tfrecord'.format(datatype) % ifile
        if ifile % 10 == 0:
            print(frame_lvl_record)
    
        tf_dataset = tf.data.TFRecordDataset(frame_lvl_record)
        tf_dataset = tf_dataset.map(parser)
        iterator = tf.compat.v1.data.make_one_shot_iterator(tf_dataset)
        next_element = iterator.get_next()
        try:
            while True:
                data_record = sess.run(next_element)
                dataset = dict()
                dataset['video_id'] = data_record[0].decode()
                dataset['video_labels'] = list()
                video_labels_list = list(data_record[1].values)
                dataset['segment_start_times'] = list(data_record[2].values)
                dataset['segment_end_times'] = list(data_record[3].values)
                dataset['segment_labels'] = list(data_record[4].values)
                dataset['segment_scores'] = list(data_record[5].values)
                dataset['frame_rgb'] = list(data_record[6])
                dataset['frame_audio'] = list(data_record[7])
                #for i, segment_label in enumerate(dataset['segment_labels']):
                #    dataset['segment_labels'][i] = vocab_label2idx_dict[segment_label] 
                #for i, video_label in enumerate(video_labels_list):
                #    if video_label in vocab_label2idx_dict:
                #        video_idx = vocab_label2idx_dict[video_label]
                #        if video_idx in dataset['segment_labels']:
                #            dataset['video_labels'].append(video_idx)
                #np.save(out_dir + dataset['video_id'] + '.npy', np.array(dataset))
        except:
            pass
