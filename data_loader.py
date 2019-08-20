import os
import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import random

class YouTubeDataset(data.Dataset):

    def __init__(self, input_dir,
                 which_challenge, phase,
                 max_frame_length=300, max_video_length=1,
                 rgb_feature_size=1024, audio_feature_size=128,
                 num_classes=1001):
        self.input_dir = input_dir + which_challenge + \
            '/{}'.format(phase if which_challenge == '2nd_challenge' else 'valid') + '/'
        self.df = pd.read_csv(input_dir + which_challenge + '/' + phase + '.csv')
        self.max_frame_length = max_frame_length
        self.max_video_length = max_video_length
        self.rgb_feature_size = rgb_feature_size
        self.audio_feature_size = audio_feature_size
        self.num_classes = num_classes
        self.load_labels = True if phase is not 'test' else False

    def __getitem__(self, idx):
        data = np.load(self.input_dir + self.df['id'][idx], allow_pickle=True).item()
        frame_rgb = torch.Tensor(data['frame_rgb'][:self.max_frame_length])
        frame_audio = torch.Tensor(data['frame_audio'][:self.max_frame_length])

        if self.load_labels == True:
            video_label = random.sample(data['video_labels'], len(data['video_labels']))
            video_label = torch.tensor(video_label)

        return (frame_rgb, frame_audio, video_label, self.max_frame_length, self.max_video_length)

    def __len__(self):
        return len(self.df)


def collate_fn(data):
    """
    Create mini-batch tensors from the list of tuples.
    tuple = (frame_rgb, frame_audio, video_label, max_frame_length, max_video_length).

    We should build custom collate_fn rather than using default collate_fn,
    because merging frame_rgb, frame_audio and video_label (including padding) is not supported in default.

    Args:
    data: list of tuple (frame_rgb, frame_audio, video_label, max_frame_length, max_video_length).
        - frame_rgb: torch tensor of shape (variable_length, rgb_feature_size=1024).
        - frame_audio: torch tensor of shape (variable_length, audio_feature_size=128).
        - video_label: torch tensor of shape (variable_length).
        - max_frame_length: torch tensor of shape (1).
        - max_video_length: torch tensor of shape (1).

    Returns:
        - padded_frame_rgbs: torch tensor of shape (batch_size, max_frame_length, rgb_feature_size=1024).
        - padded_frame_audios: torch tensor of shape (batch_size, max_frame_length, audio_feature_size=128).
        - padded_video_labels: torch tensor of shape (batch_size, max_video_length + 1).
    """

    # Sort a data list by video_label length (descending order).
    data.sort(key=lambda x: len(x[2]), reverse=True)
    
    # frame_rgbs:   tuple of frame_rgb
    # frame_audios: tuple of frame_audio
    # video_labels: tuple of video_label
    # max_frame_lengths: tuple of max_frame_length
    # max_video_lengths: tuple of max_video_length
    frame_rgbs, frame_audios, video_labels, max_frame_lengths, max_video_lengths = zip(*data)

    batch_size = len(frame_rgbs)
    max_frame_len = max_frame_lengths[0]
    max_video_len = max_video_lengths[0]
    frame_lengths = [len(frame_rgb) for frame_rgb in frame_rgbs]
    video_lengths = [len(video_label) for video_label in video_labels]
    rgb_feature_size = frame_rgbs[0].size(1)
    audio_feature_size = frame_audios[0].size(1)

    padded_frame_rgbs = torch.zeros(batch_size, max_frame_len, rgb_feature_size)
    padded_frame_audios = torch.zeros(batch_size, max_frame_len, audio_feature_size)
    padded_video_labels = torch.zeros((batch_size, max_video_len), dtype=torch.int64)

    # Merge frame_rgbs, frame_audios, video_labels in a mini-batch
    for i, frame_rgb in enumerate(frame_rgbs):
        end = frame_lengths[i]
        padded_frame_rgbs[i, :end] = frame_rgb

    for i, frame_audio in enumerate(frame_audios):
        end = frame_lengths[i]
        padded_frame_audios[i, :end] = frame_audio
        
    for i, video_label in enumerate(video_labels):
        end = video_lengths[i]
        padded_video_labels[i, :end] = video_label

    return padded_frame_rgbs, padded_frame_audios, padded_video_labels


def get_dataloader(
    input_dir,
    which_challenge,
    phases,
    max_frame_length,
    max_video_length,
    rgb_feature_size,
    audio_feature_size,
    num_classes,
    batch_size,
    num_workers):

    youtube_datasets = {
        phase: YouTubeDataset(
            input_dir=input_dir,
            which_challenge=which_challenge,
            phase=phase,
            max_frame_length=max_frame_length,
            max_video_length=max_video_length,
            rgb_feature_size=rgb_feature_size,
            audio_feature_size=audio_feature_size,
            num_classes=num_classes)
        for phase in phases}

    data_loaders = {
        phase: torch.utils.data.DataLoader(
            dataset=youtube_datasets[phase],
            batch_size=batch_size,
            shuffle=True if phase is not 'test' else False,
            num_workers=num_workers,
            collate_fn=collate_fn)
        for phase in phases}

    dataset_sizes = {phase: len(youtube_datasets[phase]) for phase in phases}

    return data_loaders, dataset_sizes
