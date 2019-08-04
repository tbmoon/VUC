import os
import numpy as np
import pandas as pd
import glob
import argparse
import tensorflow as tf


def main(args):


    def parser(record):
        context_features = {
            'id': tf.io.FixedLenFeature([], tf.string),
            'labels': tf.io.VarLenFeature(tf.int64),
            'segment_start_times': tf.io.VarLenFeature(tf.int64),
            'segment_end_times': tf.io.VarLenFeature(tf.int64),
            'segment_labels': tf.io.VarLenFeature(tf.int64),
            'segment_scores': tf.io.VarLenFeature(tf.float32)}
        sequence_features = {
            'rgb': tf.io.FixedLenSequenceFeature([], tf.string),
            'audio': tf.io.FixedLenSequenceFeature([], tf.string)}
    
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


    input_dir = args.base_dir + 'tfrecord_datasets/{}/frame/'.format(args.which_challenge)
    output_dir = args.base_dir + 'pickled_datasets/{}/{}/'.format(args.which_challenge, args.data_type)

    os.makedirs(output_dir, exist_ok=True)
    file_paths = glob.glob(input_dir + '{}*.tfrecord'.format(args.data_type))
    
    with tf.compat.v1.Session() as sess:
        for ifile in range(args.start, args.end):
            assert(ifile < len(file_paths))
            
            frame_lvl_record = input_dir + '{}%04d.tfrecord'.format(args.data_type) % ifile
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
                    dataset['video_labels'] = list(data_record[1].values)
                    dataset['segment_start_times'] = list(data_record[2].values)
                    dataset['segment_end_times'] = list(data_record[3].values)
                    dataset['segment_labels'] = list(data_record[4].values)
                    dataset['segment_scores'] = list(data_record[5].values)
                    dataset['frame_rgb'] = list(data_record[6])
                    dataset['frame_audio'] = list(data_record[7])
                    np.save(output_dir + dataset['video_id'] + '.npy', np.array(dataset))
            except:
                pass


if __name__ == '__main__':
    '''
    Purpose:
        - tfrecord to be converted into npy format for pytorch running.
    '''
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', type=str, default='/run/media/hoosiki/WareHouse3/mtb/datasets/VU/',
                        help='base directory for input and output files.')

    parser.add_argument('--data_type', type=str, default='train',
                        help='should be selected from "train", "validate", "test".')

    parser.add_argument('--which_challenge', type=str, default='2nd_challenge',
                        help='should be selected from "2nd_challenge", "3rd_challenge".')

    parser.add_argument('--start', type=int, default=0,
                        help='should be selected from 0 to 3843. #files = 3844')

    parser.add_argument('--end', type=int, default=1,
                        help='should be selected from 1 to 3844. #files = 3844')

    args = parser.parse_args()

    main(args)
