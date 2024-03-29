import os
import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from skimage import io, transform


class YouTubeDataset(data.Dataset):

    def __init__(self,
                 input_dir,
                 which_challenge,
                 phase,
                 max_frame_length=300,
                 max_vid_label_length=5,
                 max_seg_label_length=15,
                 rgb_feature_size=1024,
                 audio_feature_size=128):
        self.input_dir = os.path.join(
            input_dir,
            which_challenge,
            '{}'.format('valid' if which_challenge == '3rd_challenge' and phase == 'train' else phase))
        self.df = pd.read_csv(os.path.join(input_dir, which_challenge, phase + '.csv'))
        self.pca = np.sqrt(np.load(os.path.join(input_dir, '../yt8m_pca/eigenvals.npy'))[:1024, 0]) + 1e-4
        self.which_challenge=which_challenge
        self.max_frame_length = max_frame_length
        self.max_vid_label_length = max_vid_label_length
        self.max_seg_label_length = max_seg_label_length
        self.rgb_feature_size = rgb_feature_size
        self.audio_feature_size = audio_feature_size
        self.load_labels = True if phase is not 'test' else False

    def __getitem__(self, idx):
        vid_id = self.df['id'][idx]
        data = torch.load(os.path.join(self.input_dir, vid_id + '.pt'))
        frame_rgb = data['frame_rgb'][:self.max_frame_length].float()
        frame_audio = data['frame_audio'][:self.max_frame_length].float()

        # referred to 'https://github.com/linrongc/youtube-8m' for PCA.
        offset = 4./512
        frame_rgb = frame_rgb - offset
        frame_rgb = frame_rgb * torch.from_numpy(self.pca)

        vid_label = torch.LongTensor([0])
        seg_label = torch.LongTensor([0])
        seg_time = torch.LongTensor([0])
        if self.load_labels == True:
            vid_label = data['video_labels']
            if self.which_challenge == '3rd_challenge':
                vid_label = data['video_segment_labels']
                seg_label = data['segment_labels']
                seg_time = data['segment_times']

        return (vid_id,
                frame_rgb,
                frame_audio,
                vid_label,
                seg_label,
                seg_time,
                self.max_frame_length,
                self.max_vid_label_length,
                self.max_seg_label_length)

    def __len__(self):
        return len(self.df)


def collate_fn(data):
    """
    Create mini-batch tensors from the list of tuples.
    tuple = (vid_id, frame_rgb, frame_audio, 
             vid_label, seg_label, seg_time,
             max_frame_length, max_vid_label_length, max_seg_label_length).

    We should build custom collate_fn rather than using default collate_fn,
    because merging frame_rgb, frame_audio, vid_label, seg_label, seg_time
    (including padding) is not supported in default.

    Args:
    data: list of tuple (vid_id, frame_rgb, frame_audio,
                         vid_label, seg_label, seg_time,
                         max_frame_length, max_vid_label_length, max_seg_label_length).
        - vid_id: list of shape(4).
        - frame_rgb: torch tensor of shape (variable_length, rgb_feature_size=1024).
        - frame_audio: torch tensor of shape (variable_length, audio_feature_size=128).
        - vid_label: torch tensor of shape (variable_length).
        - seg_label: torch tensor of shape (variable_length).
        - seg_time: torch tensor of shape (variable_length).
        - max_frame_length: torch tensor of shape (1).
        - max_vid_label_length: torch tensor of shape (1).
        - max_seg_label_length: torch tensor of shape (1).

    Returns:
        - vid_ids: list of shape (batch_size, 4). 
        - frame_lengths: torch tensor of shape (batch_size).
        - padded_frame_rgbs: torch tensor of shape (batch_size, max_frame_length, rgb_feature_size=1024).
        - padded_frame_audios: torch tensor of shape (batch_size, max_frame_length, audio_feature_size=128).
        - padded_vid_labels: torch tensor of shape (batch_size, max_vid_label_length).
        - padded_seg_labels: torch tensor of shape (batch_size, max_seg_label_length).
        - padded_seg_times: torch tensor of shape (batch_size, max_seg_label_length).
    """
    # Sort a data list by vid_label length (descending order).
    data.sort(key=lambda x: len(x[3]), reverse=True)

    # vid_ids: list of vid_id.
    # frame_rgbs: tuple of frame_rgb.
    # frame_audios: tuple of frame_audio.
    # vid_labels: tuple of vid_label.
    # seg_labels: tuple of seg_label.
    # seg_times: tuple of seg_time.
    # max_frame_lengths: tuple of max_frame_length.
    # max_vid_label_lengths: tuple of max_vid_label_length.
    # max_seg_label_lengths: tuple of max_seg_label_length.
    vid_idc, frame_rgbs, frame_audios, \
    vid_labels, seg_labels, seg_times, \
    max_frame_lengths, max_vid_label_lengths, max_seg_label_lengths = zip(*data)

    batch_size = len(frame_rgbs)
    max_frame_len = max_frame_lengths[0]
    max_vid_label_len = max_vid_label_lengths[0]
    max_seg_label_len = max_seg_label_lengths[0]
    frame_lengths = [len(frame_rgb) for frame_rgb in frame_rgbs]
    vid_label_lengths = [len(vid_label) for vid_label in vid_labels]
    seg_label_lengths = [len(seg_label) for seg_label in seg_labels]
    rgb_feature_size = frame_rgbs[0].size(1)
    audio_feature_size = frame_audios[0].size(1)

    padded_frame_rgbs = torch.zeros(batch_size, max_frame_len, rgb_feature_size)
    padded_frame_audios = torch.zeros(batch_size, max_frame_len, audio_feature_size)
    padded_vid_labels = torch.zeros((batch_size, max_vid_label_len), dtype=torch.int64)
    padded_seg_labels = torch.zeros((batch_size, max_seg_label_len), dtype=torch.int64)
    padded_seg_times = torch.zeros((batch_size, max_seg_label_len), dtype=torch.int64)

    # Merge frame_rgbs, frame_audios, vid_labels, seg_labels and seg_times in a mini-batch.
    for i, frame_rgb in enumerate(frame_rgbs):
        end = frame_lengths[i]
        padded_frame_rgbs[i, :end] = frame_rgb

    for i, frame_audio in enumerate(frame_audios):
        end = frame_lengths[i]
        padded_frame_audios[i, :end] = frame_audio

    for i, vid_label in enumerate(vid_labels):
        end = vid_label_lengths[i]
        padded_vid_labels[i, :end] = vid_label

    for i, seg_label in enumerate(seg_labels):
        end = seg_label_lengths[i]
        padded_seg_labels[i, :end] = seg_label

    for i, seg_time in enumerate(seg_times):
        end = seg_label_lengths[i]
        padded_seg_times[i, :end] = seg_time

    frame_lengths = torch.LongTensor(frame_lengths)

    return vid_idc, frame_lengths, padded_frame_rgbs, \
           padded_frame_audios, padded_vid_labels, \
           padded_seg_labels, padded_seg_times


def get_dataloader(
    input_dir,
    which_challenge,
    phases,
    max_frame_length,
    max_vid_label_length,
    max_seg_label_length,
    rgb_feature_size,
    audio_feature_size,
    batch_size,
    num_workers):

    youtube_datasets = {
        phase: YouTubeDataset(
            input_dir=input_dir,
            which_challenge=which_challenge,
            phase=phase,
            max_frame_length=max_frame_length,
            max_vid_label_length=max_vid_label_length,
            max_seg_label_length=max_seg_label_length,
            rgb_feature_size=rgb_feature_size,
            audio_feature_size=audio_feature_size)
        for phase in phases}

    data_loaders = {
        phase: torch.utils.data.DataLoader(
            dataset=youtube_datasets[phase],
            batch_size=batch_size,
            shuffle=True if phase is not 'test' else False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True)
        for phase in phases}

    return data_loaders
