import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


class YouTubeDataset(data.Dataset):

    def __init__(self, input_dir, phase, num_seg_frames=5, max_frame_length=300, rgb_feature_size=1024, audio_feature_size=128):
        self.input_dir = input_dir + 'npy_formatted_frame/{}/'.format('validate' if phase is not 'test' else 'test')
        self.df = pd.read_csv(input_dir + phase + '.csv')
        self.num_seg_frames = num_seg_frames
        self.max_frame_length = max_frame_length
        self.rgb_feature_size = rgb_feature_size
        self.audio_feature_size = audio_feature_size
        self.load_labels = True if phase is not 'test' else False

    def __getitem__(self, idx):
        data = np.load(self.input_dir + self.df['id'][idx], allow_pickle=True).item()
        frame_rgb = data['frame_rgb']
        frame_audio = data['frame_audio']

        '''
        Keep multiples of num_seg_frames (=5) of the frames for a video.
        '''
        frame_length = len(frame_rgb) - (len(frame_rgb) % self.num_seg_frames)
        padded_frame_rgb = np.array([np.array([0.] * self.rgb_feature_size)] * self.max_frame_length)
        padded_frame_rgb[:frame_length] = frame_rgb[:frame_length]
        padded_frame_audio = np.array([np.array([0.] * self.audio_feature_size)] * self.max_frame_length)
        padded_frame_audio[:frame_length] = frame_audio[:frame_length]

        sample = {
            'frame_length': frame_length,
            'frame_rgb': padded_frame_rgb,
            'frame_audio': padded_frame_audio}

        if self.load_labels == True:
            segment_labels = np.array(data['segment_labels'])
            segment_scores = np.array(data['segment_scores'])
            segment_start_times = np.array(data['segment_start_times'])

            segment_labels = segment_labels * segment_scores
            segment_labels = [int(i) for i in segment_labels]

            video_labels = np.array(data['video_labels'])
            video_labels = np.random.choice(video_labels) if len(video_labels) != 0 else np.random.randint(1001)

            padded_segment_labels = np.array(
                [0] * (self.max_frame_length // self.num_seg_frames + 
                   (0 if self.max_frame_length % self.num_seg_frames == 0 else 1)))
            padded_segment_labels[segment_start_times // self.num_seg_frames] = segment_labels

            sample['video_labels'] = video_labels
            sample['segment_labels'] = padded_segment_labels

        return sample

    def __len__(self):
        return len(self.df)


def get_dataloader(
    input_dir,
    phases,
    num_seg_frames,
    max_frame_length,
    rgb_feature_size,
    audio_feature_size,
    batch_size,
    num_workers):

    youtube_datasets = {
        phase: YouTubeDataset(
            input_dir=input_dir,
            phase=phase,
            num_seg_frames=num_seg_frames,
            max_frame_length=max_frame_length,
            rgb_feature_size=rgb_feature_size,
            audio_feature_size=audio_feature_size)
        for phase in phases}

    data_loaders = {
        phase: torch.utils.data.DataLoader(
            dataset=youtube_datasets[phase],
            batch_size=batch_size,
            shuffle=True if phase is not 'test' else False,
            num_workers=num_workers,
            drop_last=True if phase is not 'test' else False)
        for phase in phases}

    dataset_sizes = {phase: len(youtube_datasets[phase]) for phase in phases}

    return data_loaders, dataset_sizes
