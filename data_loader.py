import os
import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


class YouTubeDataset(data.Dataset):

    def __init__(self, input_dir, phase, max_frame_length=301, rgb_feature_size=1024, audio_feature_size=128):
        self.input_dir = input_dir + 'npy_formatted_frame/{}/'.format('validate' if phase is not 'test' else 'test')
        self.df = pd.read_csv(input_dir + phase + '.csv')
        self.max_frame_length = max_frame_length
        self.rgb_feature_size = rgb_feature_size
        self.audio_feature_size = audio_feature_size
        self.load_labels = True if phase is not 'test' else False

    def __getitem__(self, idx):
        data = np.load(self.input_dir + self.df['id'][idx], allow_pickle=True).item()
        frame_rgb = torch.Tensor(data['frame_rgb'])
        frame_audio = torch.Tensor(data['frame_audio'])

        if self.load_labels == True:
            video_label = np.array(data['video_labels'])
            video_label = random.choice(video_label)
            video_label = torch.tensor(video_label)
            
        return (frame_rgb, frame_audio, video_label)

    def __len__(self):
        return len(self.df)


def collate_fn(data):
    """
    Create mini-batch tensors from the list of tuples (frame_rgb, frame_audio, video_label).

    We should build custom collate_fn rather than using default collate_fn,
    because merging frame_rgb and frame_audio (including padding) is not supported in default.

    Args:
    data: list of tuple (frame_rgb, frame_audio, video_label).
        - frame_rgb: torch tensor of shape (variable_length, rgb_feature_size=1024).
        - frame_audio: torch tensor of shape (variable_length, audio_feature_size=128).
        - video_label: torch tensor of shape (1).

    Returns:
        - padded_frame_rgbs: torch tensor of shape (batch_size, padded_length, rgb_feature_size=1024).
        - padded_frame_audios: torch tensor of shape (batch_size, padded_length, audio_feature_size=128).
        - video_labels: torch tensor of shape (batch_size, 1) 
    """
    
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # frame_rgbs:   tuple of frame_rgb
    # frame_audios: tuple of frame_audio
    # video_labels: tuple of video_label
    frame_rgbs, frame_audios, video_labels = zip(*data)

    batch_size = len(frame_rgbs)
    frame_lengths = [len(frame_rgb) for frame_rgb in frame_rgbs]
    max_frame_len = 301
    rgb_feature_size = frame_rgbs[0].size(1)
    audio_feature_size = frame_audios[0].size(1)

    padded_frame_rgbs = torch.zeros(batch_size, max_frame_len, rgb_feature_size)
    padded_frame_audios = torch.zeros(batch_size, max_frame_len, audio_feature_size)

    # Merge frame_rgbs and frame_audio in a mini-batch
    for i, frame_rgb in enumerate(frame_rgbs):
        end = frame_lengths[i]
        padded_frame_rgbs[i, :end] = frame_rgb

    for i, frame_audio in enumerate(frame_audios):
        end = frame_lengths[i]
        padded_frame_audios[i, :end] = frame_audio

    video_labels = torch.stack(video_labels, 0)

    return padded_frame_rgbs, padded_frame_audios, video_labels


def get_dataloader(
    input_dir,
    phases,
    max_frame_length,
    rgb_feature_size,
    audio_feature_size,
    batch_size,
    num_workers):

    youtube_datasets = {
        phase: YouTubeDataset(
            input_dir=input_dir,
            phase=phase,
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
            collate_fn=collate_fn)
        for phase in phases}

    dataset_sizes = {phase: len(youtube_datasets[phase]) for phase in phases}

    return data_loaders, dataset_sizes
