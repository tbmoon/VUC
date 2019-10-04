import os
import argparse
import numpy as np
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import BaseModel, TransformerModel, TransformerModel_V2


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def main(args):

    output_dir = os.path.join(os.getcwd(), 'tests_tmp')
    os.makedirs(output_dir, exist_ok=True)

    model = BaseModel(
        rgb_feature_size=args.rgb_feature_size,
        audio_feature_size=args.audio_feature_size,
        d_l=args.d_l,
        num_classes=args.num_classes)
    model = model.to(device)

    checkpoint = torch.load(os.path.join(os.getcwd(), 'models/model-epoch-pretrained-base-finetune.ckpt'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    pca = np.sqrt(np.load(os.path.join(args.input_dir, '../../yt8m_pca/eigenvals.npy'))[:1024, 0]) + 1e-4

    df_input = pd.read_csv(os.path.join(args.input_dir, 'test.csv'))

    for i in range(1, args.num_classes+1):
        df_output = pd.DataFrame(columns=['vid_id', 'vid_label_pred', 'vid_prob'])
        df_output.to_csv(os.path.join(output_dir, '%04d.csv'%i), index=False)

    for idx, vid_id in enumerate(df_input.id):
        #if idx > 5:
        #    break
        if idx % 100 == 0:
            print('idx:', idx, ' vid_id:', vid_id)
            
        data = torch.load(os.path.join(args.input_dir, 'test', vid_id + '.pt'))
        vid_labels = data['video_labels']
        seg_labels = data['segment_labels']
        seg_times = data['segment_times']
        frame_rgb = data['frame_rgb'][:args.max_frame_length].float()
        frame_audio = data['frame_audio'][:args.max_frame_length].float()

        # referred to 'https://github.com/linrongc/youtube-8m' for PCA.
        offset = 4./512
        frame_rgb = frame_rgb - offset
        frame_rgb = frame_rgb * torch.from_numpy(pca)

        frame_len = frame_rgb.size(0)
        frame_rgbs = torch.zeros(1, args.max_frame_length, args.rgb_feature_size)
        frame_audios = torch.zeros(1, args.max_frame_length, args.audio_feature_size)
        frame_rgbs[0, :frame_len] = frame_rgb      # [1, max_frame_length, rgb_feature_size]
        frame_audios[0, :frame_len] = frame_audio  # [1, max_frame_length, audio_feature_size]
        frame_rgbs = frame_rgbs.to(device)
        frame_audios = frame_audios.to(device)

        # vid_probs: [batch_size, num_classes]
        # attn_idc: [batch_size, num_classes]
        # scores: [batch_size, seg_length, n_attns]
        vid_probs, attn_weights = model(frame_rgbs, frame_audios)

        # vid_probs: [num_classes]
        vid_probs = vid_probs.squeeze(0).cpu()

        # vid_probs: [max_vid_label_length]
        # vid_label_preds: [max_vid_label_length]
        vid_probs, vid_label_preds = torch.topk(vid_probs, args.max_vid_label_length)

        vid_label_preds = vid_label_preds + 1
        vid_label_preds = vid_label_preds.numpy()
        vid_probs = vid_probs.cpu().detach().numpy()

        for i in range(args.max_vid_label_length):
            df_output = pd.read_csv(os.path.join(output_dir, '%04d.csv'%vid_label_preds[i]))
            df_output = df_output.append({'vid_id': vid_id,
                                          'vid_label_pred': vid_label_preds[i],
                                          'vid_prob': vid_probs[i]}, ignore_index=True)
            df_output.to_csv(os.path.join(output_dir, '%04d.csv'%vid_label_preds[i]), index=False)


        if args.print == True:
            print("vid_id", vid_id)
            print("vid_labels:", vid_labels.numpy())
            print("vid_label_preds:", vid_label_preds, "\n")
            print("vid_probs:", vid_probs, "\n")
            print()
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str,
                        default='/run/media/hoosiki/WareHouse2/mtb/datasets/VU/pytorch_datasets/3rd_challenge',
                        help='input directory for video understanding challenge.')

    parser.add_argument('--print', type=bool, default=False,
                        help='print.')

    parser.add_argument('--model_name', type=str, default='base',
                        help='transformer, base.')

    parser.add_argument('--max_frame_length', type=int, default=300,
                        help='the maximum length of frame. (301)')

    parser.add_argument('--max_seg_length', type=int, default=60,
                        help='the maximum length of segment step. (60)')

    parser.add_argument('--max_vid_label_length', type=int, default=20,
                        help='the maximum length of video label for 3rd challenge. (4)')
    
    parser.add_argument('--max_seg_label_length', type=int, default=5,
                        help='the maximum length of segment label for 3rd challenge. (5)')

    parser.add_argument('--n_layers', type=int, default=2,
                        help='n_layers for the encoder. (6)')

    parser.add_argument('--n_heads', type=int, default=4,
                        help='n_heads for the attention. (8)')

    parser.add_argument('--rgb_feature_size', type=int, default=1024,
                        help='rgb feature size in a frame. (1024)')

    parser.add_argument('--audio_feature_size', type=int, default=128,
                        help='audio feature size in a frame. (128)')

    parser.add_argument('--d_rgb', type=int, default=2048,
                         help='mapping rgb size. (2048)')

    parser.add_argument('--d_audio', type=int, default=256,
                         help='mapping audio size. (256)')

    parser.add_argument('--d_model', type=int, default=128,
                        help='d_model for feature projection. \
                              512 for paper. (256)')

    parser.add_argument('--d_l', type=int, default=256,
                        help='d_l for q, k, v projection. (64)')
    
    parser.add_argument('--d_proj', type=int, default=1024,
                        help='d_proj for q, k, v projection. (64)')

    parser.add_argument('--d_ff', type=int, default=256,
                        help='d_ff. 2048 for paper. (1024)')

    parser.add_argument('--d_linear', type=int, default=512,
                        help='d_linear. (2048)')

    parser.add_argument('--n_attns', type=int, default=4,
                        help='n_heads for the attention. (4)')

    parser.add_argument('--num_classes', type=int, default=1000,
                        help='the number of classes. (1000) / (3862)')

    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout. (0.1)')

    args = parser.parse_args()

    main(args)
