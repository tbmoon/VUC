import os
import argparse
import numpy as np
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data_loader import YouTubeDataset, get_dataloader
from models import BaseModel, TransformerModel, TransformerModel_V2


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def main(args):

    since = time.time()
    output_dir = os.path.join(os.getcwd(), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    data_loaders = get_dataloader(
        input_dir=args.input_dir,
        which_challenge='3rd_challenge',
        phases=['test'],
        max_frame_length=args.max_frame_length,
        max_vid_label_length=args.max_vid_label_length,
        max_seg_label_length=args.max_seg_label_length,
        rgb_feature_size=args.rgb_feature_size,
        audio_feature_size=args.audio_feature_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    model = BaseModel(
        rgb_feature_size=args.rgb_feature_size,
        audio_feature_size=args.audio_feature_size,
        d_rgb=args.d_rgb,
        d_audio=args.d_audio,
        d_l=args.d_l,
        num_classes=args.num_classes,
        dropout=args.dropout)
    model = model.to(device)

    checkpoint = torch.load(os.path.join(os.getcwd(), 'models/model-epoch-pretrained-base-finetune.ckpt'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    df_outputs = {i: pd.DataFrame(columns=['vid_id', 'vid_label_pred', 'vid_prob', 'seg_label_pred', 'seg_prob']) \
                      for i in range(1, args.num_classes+1)}

    for idx, (vid_ids, frame_lengths, frame_rgbs, frame_audios, vid_labels, seg_labels, seg_times) \
        in enumerate(data_loaders['test']):           

        if idx%10 == 0:
            print('idx:', idx)

        # frame_rgbs: [batch_size, frame_length, rgb_feature_size]
        # frame_audios: [batch_size, frame_length, audio_feature_size]
        frame_rgbs = frame_rgbs.to(device)
        frame_audios = frame_audios.to(device)
        batch_size = frame_audios.size(0)

        # vid_probs: [batch_size, num_classes]
        # attn_weights: [batch_size, max_seg_length]
        vid_probs, attn_weights = model(frame_rgbs, frame_audios)

        # vid_probs: [batch_size, max_vid_label_length]
        # vid_label_preds: [batch_size, max_vid_label_length]
        vid_probs, vid_label_preds = torch.topk(vid_probs, args.max_vid_label_length)
        seg_probs, seg_label_preds = torch.topk(attn_weights, args.seg_pred_length)

        vid_probs = vid_probs.cpu().detach().numpy()
        vid_label_preds = vid_label_preds.cpu().numpy()
        vid_label_preds = vid_label_preds + 1
        seg_probs = seg_probs.cpu().detach().numpy()
        seg_label_preds = seg_label_preds.cpu().numpy()
        seg_label_preds = seg_label_preds + 1

        for i in range(batch_size):
            for j in range(args.max_vid_label_length):
                vid_label_pred = vid_label_preds[i][j]
                df_outputs[vid_label_pred] = df_outputs[vid_label_pred].append(
                    {'vid_id': vid_ids[i],
                     'vid_label_pred': vid_label_pred,
                     'vid_prob': vid_probs[i][j],
                     'seg_label_pred': seg_label_preds[i],
                     'seg_prob': seg_probs[i]}, ignore_index=True)

    for i in range(1, args.num_classes+1):
        df_outputs[i].to_csv(os.path.join(output_dir, '%04d.csv'%i), index=False)

    time_elapsed = time.time() - since
    print('=> Running time in a epoch: {:.0f}h {:.0f}m {:.0f}s'
          .format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str,
                        default='/run/media/hoosiki/WareHouse2/mtb/datasets/VU/pytorch_datasets',
                        help='input directory for video understanding challenge.')

    parser.add_argument('--seg_pred_length', type=int, default=20,
                        help='the maximum length of segment step. (60)')

    parser.add_argument('--max_frame_length', type=int, default=300,
                        help='the maximum length of frame. (301)')

    parser.add_argument('--max_seg_length', type=int, default=60,
                        help='the maximum length of segment step. (60)')

    parser.add_argument('--max_vid_label_length', type=int, default=20,
                        help='the maximum length of video label for 3rd challenge. (4)')
    
    parser.add_argument('--max_seg_label_length', type=int, default=5,
                        help='the maximum length of segment label for 3rd challenge. (5)')

    parser.add_argument('--rgb_feature_size', type=int, default=1024,
                        help='rgb feature size in a frame. (1024)')

    parser.add_argument('--audio_feature_size', type=int, default=128,
                        help='audio feature size in a frame. (128)')

    parser.add_argument('--d_rgb', type=int, default=2048,
                         help='mapping rgb size. (2048)')

    parser.add_argument('--d_audio', type=int, default=256,
                         help='mapping audio size. (256)')

    parser.add_argument('--d_l', type=int, default=256,
                        help='d_l for attentions. (256)')
    
    parser.add_argument('--num_classes', type=int, default=1000,
                        help='the number of classes. (1000)')

    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout. (0.1)')

    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch_size. (100)')

    parser.add_argument('--num_workers', type=int, default=10,
                        help='the number of processes working on cpu. (10)')

    args = parser.parse_args()

    main(args)
