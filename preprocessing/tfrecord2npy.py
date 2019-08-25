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

    assert(args.which_challenge == '2nd_challenge' or args.which_challenge == '3rd_challenge')

    input_dir = args.base_dir + 'tfrecord_datasets/{}/frame/'.format(args.which_challenge)
    output_dir = args.out_dir + 'pytorch_datasets/{}/{}/'.format(args.which_challenge, args.data_type)

    os.makedirs(output_dir, exist_ok=True)
    if args.data_type == 'train' or args.data_type == 'test':
        file_paths = glob.glob(input_dir + '{}*.tfrecord'.format(args.data_type))
    else:
        file_paths = glob.glob(input_dir + 'validate*.tfrecord')

    vocab_label2idx_dict = dict()
    if args.which_challenge == '2nd_challenge' and args.use_all_classes == 'yes':
        df_vocab = pd.read_csv('../data/2nd_challenge_vocabulary.csv')
    else:
        df_vocab = pd.read_csv('../data/3rd_challenge_vocabulary.csv')    
    for i, label in enumerate(df_vocab['Index']):
        vocab_label2idx_dict[label] = i+1

    with tf.compat.v1.Session() as sess:
        for ifile in range(args.start, args.end):
            assert(ifile < len(file_paths))

            if args.data_type == 'train' or args.data_type == 'test':
                frame_lvl_record = input_dir + '{}%04d.tfrecord'.format(args.data_type) % ifile
            else:
                frame_lvl_record = input_dir + 'validate%04d.tfrecord' % ifile
            print(frame_lvl_record)
            tf_dataset = tf.data.TFRecordDataset(frame_lvl_record)
            tf_dataset = tf_dataset.map(parser)
            iterator = tf.compat.v1.data.make_one_shot_iterator(tf_dataset)
            next_element = iterator.get_next()
            try:
                while True:
                    '''
                        - data_record[2] for segment start time.
                        - data_record[3] for segment end time.
                        - segment end time is used instead of start time due to background (index = 0).
                    '''
                    data_record = sess.run(next_element)
                    dataset = dict()
                    dataset['video_id'] = data_record[0].decode()
                    dataset['frame_rgb'] = list(data_record[6])
                    dataset['frame_audio'] = list(data_record[7])

                    raw_segment_start_times_list = list(data_record[3].values)
                    raw_segment_labels_list = list(data_record[4].values)
                    raw_segment_scores_list = list(data_record[5].values)

                    if args.which_challenge == '2nd_challenge':
                        dataset['video_labels'] = list()
                        video_labels_list = list(data_record[1].values)
                        for i, video_label in enumerate(video_labels_list):
                            if video_label in vocab_label2idx_dict:
                                video_idx = vocab_label2idx_dict[video_label]
                                dataset['video_labels'].append(video_idx)
                    else:
                        for i, segment_label in enumerate(raw_segment_labels_list):
                            raw_segment_labels_list[i] = vocab_label2idx_dict[segment_label] 

                        segment_start_times_list = \
                            [start / 5 for start, score 
                             in zip(raw_segment_start_times_list, raw_segment_scores_list) if score == 1]
                        segment_labels_list = \
                            [label for label, score 
                             in zip(raw_segment_labels_list, raw_segment_scores_list) if score == 1]
                        video_labels_list = segment_labels_list

                        dataset['segment_start_times'] = segment_start_times_list
                        dataset['segment_labels'] = segment_labels_list
                        dataset['video_labels'] = list(set(video_labels_list))

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

    parser.add_argument('--out_dir', type=str, default='/run/media/hoosiki/WareHouse2/mtb/datasets/VU/',
                        help='output directory for input and output files.')

    parser.add_argument('--data_type', type=str, default='valid',
                        help='should be selected from "train", "valid", "test".')

    parser.add_argument('--which_challenge', type=str, default='3rd_challenge',
                        help='should be selected from "2nd_challenge", "3rd_challenge".')

    parser.add_argument('--use_all_classes', type=str, default='no',
                        help='yes: use all classes in 2nd challenge. no: use classes only in 3rd challenge')

    parser.add_argument('--start', type=int, default=0,
                        help='should be selected from 0 to 3843. #files = 3844')

    parser.add_argument('--end', type=int, default=1,
                        help='should be selected from 1 to 3844. #files = 3844')

    args = parser.parse_args()

    main(args)
