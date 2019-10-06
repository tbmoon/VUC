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


# Batched index_select
def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy) # b x e x f
    return out


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

    model = TransformerModel(
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        rgb_feature_size=args.rgb_feature_size,
        audio_feature_size=args.audio_feature_size,
        d_rgb=args.d_rgb,
        d_audio=args.d_audio,
        d_model=args.d_model,
        d_ff=args.d_ff,
        d_proj=args.d_proj,
        n_attns = args.n_attns,
        num_classes=args.num_classes,
        dropout=args.dropout)
    model = model.to(device)

    checkpoint = torch.load(os.path.join(os.getcwd(), 'models/model-epoch-pretrained-transformer.ckpt'))
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
        # attn_idc: [batch_size, num_classes]
        # scores: [batch_size, max_seg_length, n_attns]
        # attn_weights: [batch_size, max_seg_length, n_attns]
        vid_probs, attn_idc, scores, attn_weights, conv_loss = model(frame_rgbs, frame_audios, device)

        # vid_probs: [batch_size, vid_pred_length]
        # vid_label_preds: [batch_size, vid_pred_length]
        vid_probs, vid_label_preds = torch.topk(vid_probs, args.vid_pred_length)
        vid_label_preds = vid_label_preds + 1

        # attn_idc: [batch_size, num_classes+1]
        zeros = torch.zeros(batch_size, 1).long().to(device)
        attn_idc = torch.cat((zeros, attn_idc), dim=1)

        # selected_attn_idc: [batch_size, vid_pred_length]
        selected_attn_idc = torch.gather(attn_idc, 1, vid_label_preds)

        # attn_weights: [batch_size, n_attns, max_seg_length]
        attn_weights = attn_weights.transpose(1, 2)

        # selected_attn_weights: [batch_size, vid_pred_length, max_seg_length]
        selected_attn_weights = batched_index_select(attn_weights, 1, selected_attn_idc)

		# seg_probs: [batch_size, seg_pred_length] 
		# seg_label_preds: [batch_size, seg_pred_length] 
        seg_probs, seg_label_preds = torch.topk(selected_attn_weights, args.seg_pred_length)
        seg_label_preds = seg_label_preds + 1

        # To save predictions, converted to numpy data.
        vid_probs = vid_probs.cpu().detach().numpy()
        vid_label_preds = vid_label_preds.cpu().numpy()
        seg_probs = seg_probs.cpu().detach().numpy()
        seg_label_preds = seg_label_preds.cpu().numpy()

        for i in range(batch_size):
            for j in range(args.max_vid_label_length):
                vid_label_pred = vid_label_preds[i][j]
                df_outputs[vid_label_pred] = df_outputs[vid_label_pred].append(
                    {'vid_id': vid_ids[i],
                     'vid_label_pred': vid_label_pred,
                     'vid_prob': vid_probs[i][j],
                     'seg_label_pred': list(seg_label_preds[i]),
                     'seg_prob': list(seg_probs[i])}, ignore_index=True)

    for i in range(1, args.num_classes+1):
        df_outputs[i].to_csv(os.path.join(output_dir, '%04d.csv'%i), index=False)

    time_elapsed = time.time() - since
    print('=> Running time in a epoch: {:.0f}h {:.0f}m {:.0f}s'
          .format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str,
                        default='/run/media/hoosiki/WareHouse1/mtb/datasets/VU/pytorch_datasets',
                        help='input directory for video understanding challenge.')

    parser.add_argument('--vid_pred_length', type=int, default=22,
                        help='the maximum length of vid prediction. (60)')

    parser.add_argument('--seg_pred_length', type=int, default=20,
                        help='the maximum length of seg prediction. (60)')

    parser.add_argument('--max_frame_length', type=int, default=300,
                        help='the maximum length of frame. (301)')

    parser.add_argument('--max_seg_length', type=int, default=60,
                        help='the maximum length of segment step. (60)')

    parser.add_argument('--max_vid_label_length', type=int, default=20,
                        help='the maximum length of video label for 3rd challenge. (4)')
    
    parser.add_argument('--max_seg_label_length', type=int, default=15,
                        help='the maximum length of segment label for 3rd challenge. (15)')

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

    parser.add_argument('--d_l', type=int, default=256,
                        help='d_l for attentions. (256)')

    parser.add_argument('--d_model', type=int, default=128,
                        help='d_model for feature projection. \
                              512 for paper. (256)')

    parser.add_argument('--d_proj', type=int, default=1024,
                        help='d_proj for q, k, v projection. (64)')

    parser.add_argument('--d_ff', type=int, default=256,
                        help='d_ff. 2048 for paper. (1024)')

    parser.add_argument('--d_linear', type=int, default=512,
                        help='d_linear. (2048)')

    parser.add_argument('--n_attns', type=int, default=4,
                        help='n_heads for the attention. (4)')
    
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
