import os
import argparse
import numpy as np
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from data_loader import YouTubeDataset, get_dataloader
from models import TransformerModel


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(args):

    os.makedirs('./outputs', exist_ok=True)

    model = TransformerModel(
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        rgb_feature_size=args.rgb_feature_size,
        audio_feature_size=args.audio_feature_size,
        d_model=args.d_model,
        d_att=args.d_att,
        d_hop=args.d_hop,
        d_ff=args.d_ff,
        num_classes=args.num_classes,
        dropout=args.dropout)
    
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    checkpoint = torch.load(args.model_dir + '/model-epoch-33.ckpt')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    df = pd.read_csv(args.input_dir + 'test.csv')
    
    df_vocab = pd.read_csv('./preprocessing/vocabulary.csv')
    vocab_idx2label_dict = dict()
    for i, label in enumerate(df_vocab['Index']):
        vocab_idx2label_dict[i+1] = label
    
    df_result = pd.DataFrame(columns=['class', 'video_id', 'start_time', 'attn'])
    
    for idx, video_id in enumerate(df.id):

        if idx % 100 == 0:
            print('idx', idx)

        data = np.load(args.input_dir + 'test/' + video_id, allow_pickle=True).item()
        frame_rgb = torch.Tensor(data['frame_rgb']).to(device)
        frame_audio = torch.Tensor(data['frame_audio']).to(device)
        frame_rgb = frame_rgb.unsqueeze(0)
        frame_audio = frame_audio.unsqueeze(0)
        frame_rgb = torch.cat((frame_rgb, frame_rgb), 0)
        frame_audio = torch.cat((frame_audio, frame_audio), 0)

        # outputs: [batch_size, num_classes = 1001]
        outputs, attns = model(frame_rgb, frame_audio)
        _, preds = torch.max(outputs, 1)

        pred = preds.cpu().detach().numpy()[0]
        attn = attns.cpu().detach().numpy()[0]
        length = attn.shape[1]
        max_attn = np.max(attn)
        start_time = np.argmax(attn) % length // 5 * 5

        df_result = df_result.append({'class': int(vocab_idx2label_dict[pred]),
                                      'video_id': video_id[:4],
                                      'start_time': int(start_time),
                                      'attn': max_attn}, ignore_index=True)        

    df_result.sort_values(by=['attn'], inplace=True, ascending=False)
    df_result.to_csv("./outputs/results.csv", index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str,
                        default='/run/media/hoosiki/WareHouse3/mtb/datasets/VU/pickled_datasets/3rd_challenge/',
                        help='input directory for video understanding challenge.')

    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='directory for logs.')

    parser.add_argument('--model_dir', type=str, default='./models',
                        help='directory for saved models.')

    parser.add_argument('--max_frame_length', type=int, default=301,
                        help='the maximum length of frame = 301.')

    parser.add_argument('--n_layers', type=int, default=6,
                        help='n_layers for the encoder.')

    parser.add_argument('--n_heads', type=int, default=8,
                        help='n_heads for the attention.')

    parser.add_argument('--rgb_feature_size', type=int, default=1024,
                        help='rgb feature size in a frame.')

    parser.add_argument('--audio_feature_size', type=int, default=128,
                        help='audio feature size in a frame.')

    parser.add_argument('--d_model', type=int, default=64,
                        help='d_model for feature projection.')
    
    parser.add_argument('--d_att', type=int, default=80,
                        help='d_att.')

    parser.add_argument('--d_hop', type=int, default=8,
                        help='d_hop.')

    parser.add_argument('--d_ff', type=int, default=128,
                        help='d_ff.')

    parser.add_argument('--num_classes', type=int, default=1001,
                        help='the number of classes. 1000+1 / 3862')

    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout.')

    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate for training.')

    parser.add_argument('--step_size', type=int, default=10,
                        help='period of learning rate decay.')

    parser.add_argument('--gamma', type=float, default=0.1,
                        help='multiplicative factor of learning rate decay.')

    parser.add_argument('--num_epochs', type=int, default=100,
                        help='the number of epochs.')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size.')

    parser.add_argument('--num_workers', type=int, default=16,
                        help='the number of processes working on cpu.')

    parser.add_argument('--save_step', type=int, default=1,
                        help='save step of model.')

    args = parser.parse_args()

    main(args)
