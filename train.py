import os
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
#from center_loss import CenterLoss
from data_loader import YouTubeDataset, get_dataloader
from models import BaseModel, TransformerModel, TransformerModel_V2


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def video_label_loss(probs, labels):
    '''
    inputs:
        - probs: [batch_size, num_classes]
        - labels: [batch_size, max_vid_label_length]
    outputs:
        - loss: [1]
    '''
    eps = 1e-6
    batch_size = probs.size(0)
    num_classes = probs.size(1)

    # is_selected: [batch_size, num_classes+1]
    is_selected = torch.zeros(batch_size, num_classes+1).to(device).scatter(1, labels, 1).long()
    # is_selected: [batch_size, num_classes]
    is_selected = is_selected[:, 1:]

    # probs: [batch_size, num_classes]
    probs = torch.where(is_selected == 1, probs, 1 - probs)

    loss = -1 * (1 - probs)**args.focal_loss_gamma * torch.log(probs + eps)
    # If you want to use balanced loss, comment in below.
    #unselected_weight = is_selected.sum(dim=1).view(-1, 1).float()  # unselected_weight: [batch_size, 1]
    #selected_weight = num_classes - unselected_weight               # selected_weight: [batch_size, 1]
    #loss = torch.where(is_selected == 1, selected_weight * loss, unselected_weight * loss)
    loss = loss.sum()
    return loss


def binary_cross_entropy_loss_with_seg_label_processing(frame_lengths, selected_vid_label, prob, seg_labels, seg_times):
    '''
    inputs:
        - frame_lengths: [batch_size]
        - selected_vid_label: [batch_size]
        - prob: [batch_size, max_seg_length]
        - seg_labels: [batch_size, max_seg_label_length]
        - seg_times: [batch_size, max_seg_label_legnth]
    outputs:
        - loss: [1]
        - label_size: [1]
    '''
    eps = 1e-6
    batch_size = prob.size(0)
    max_seg_length = prob.size(1)
    seg_lengths = frame_lengths // 5

    selected_seg_label = selected_vid_label.view(-1, 1) == seg_labels
    selected_seg_label = selected_seg_label.float() * seg_times.float()
    selected_seg_label = selected_seg_label.long()

    # segment index start from 1, not from 0.
    selected_seg_label = torch.zeros(batch_size, max_seg_length + 1).to(device).scatter_(1, selected_seg_label, 1)
    selected_seg_label = selected_seg_label[:, 1:]

    s_loss = -(selected_seg_label*torch.log(prob + eps) + (1-selected_seg_label)*torch.log(1-prob + eps))

    mask = torch.arange(max_seg_length).to(device).unsqueeze(0) < seg_lengths.unsqueeze(1)
    mask_vid_label = selected_vid_label.view(-1, 1).float().le(0.5)
    mask = torch.masked_fill(mask, mask_vid_label, 0)

    loss = s_loss.masked_select(mask).sum() 
    label_size = mask.sum()

    return loss, label_size


def main(args):

    os.makedirs(os.path.join(os.getcwd(), 'logs'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'models'), exist_ok=True)

    # the maximum length of video label in 2nd challenge: 18.
    # the maximum length of video label in 3rd challenge: 4.
    args.max_vid_label_length = 18 if args.which_challenge == '2nd_challenge' else 4

    data_loaders, dataset_sizes = get_dataloader(
        input_dir=args.input_dir,
        which_challenge=args.which_challenge,
        phases=['train', 'valid'],
        max_frame_length=args.max_frame_length,
        max_vid_label_length=args.max_vid_label_length,
        max_seg_label_length=args.max_seg_label_length,
        rgb_feature_size=args.rgb_feature_size,
        audio_feature_size=args.audio_feature_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.model_name == 'transformer':
        model = TransformerModel_V2(
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
    elif args.model_name == 'base':
        model = BaseModel(
            rgb_feature_size=args.rgb_feature_size,
            audio_feature_size=args.audio_feature_size,
            num_classes=args.num_classes)
    model = model.to(device)

    #center_loss = CenterLoss(num_classes=args.num_classes, feat_dim=args.d_model, device=device)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if args.load_model == True:
        checkpoint = torch.load(os.path.join(os.getcwd(), 'models/model-epoch-pretrained.ckpt'))
        model.load_state_dict(checkpoint['state_dict'])

    params = list(model.parameters()) #+ list(center_loss.parameters())
    optimizer = optim.Adam(params, lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    for epoch in range(args.num_epochs):
        for phase in ['train', 'valid']:
            since = time.time()
            running_vid_label_loss = 0.0
            running_conv_loss = 0.0
            running_vid_cent_loss = 0.0
            running_time_loss = 0.0
            running_vid_label_corrects = 0
            running_vid_label_size = 0
            running_conv_size = 0
            running_time_label_size = 0
            running_num_vid_labels = 0

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for idx, (frame_lengths, frame_rgbs, frame_audios, vid_labels, seg_labels, seg_times) \
                in enumerate(data_loaders[phase]):
                
                optimizer.zero_grad()

                # frame_lengths: [batch_size]
                # frame_rgbs: [batch_size, max_frame_length=300, rgb_feature_size=1024]
                # frame_audios: [batch_size, max_frame_length=300, audio_feature_size=128]
                # vid_labels: [batch_size, max_vid_label_length]
                # seg_labels: [batch_size, max_seg_label_length]
                # seg_times: [batch_size, max_seg_label_length]
                frame_lengths = frame_lengths.to(device)
                frame_rgbs = frame_rgbs.to(device)
                frame_audios = frame_audios.to(device)
                vid_labels = vid_labels.to(device)
                seg_labels = seg_labels.to(device)
                seg_times = seg_times.to(device)
                batch_size = vid_labels.size(0)
                vid_label_size = args.num_classes * batch_size
                time_label_size = 0
                vid_label_lengths = vid_labels.float().ge(0.5).sum(dim=1).view(-1, 1)
                num_vid_labels = vid_labels.float().ge(0.5).sum()

                with torch.set_grad_enabled(phase == 'train'):
                    total_loss = 0.0
                    vid_label_loss = 0.0
                    vid_cent_loss = 0.0
                    time_loss = 0.0

                    # vid_probs: [batch_size, num_classes]
                    # attn_idc: [batch_size, num_classes]
                    # attn_weights: [batch_size, seg_length, n_attns]
                    # conv_loss: []
                    if args.model_name == 'transformer':
                        vid_probs, attn_idc, scores, attn_weights, conv_loss = model(frame_rgbs, frame_audios, device)
                    elif args.model_name == 'base':
                        vid_probs = model(frame_rgbs, frame_audios)
                    
                    vid_label_loss = video_label_loss(vid_probs, vid_labels)

                    _, vid_preds = torch.topk(vid_probs, args.max_vid_label_length)
                    vid_preds = vid_preds + 1
                    mask = torch.arange(1, args.max_vid_label_length+1).to(device)
                    mask = mask.expand(batch_size, args.max_vid_label_length)
                    zeros = torch.zeros(batch_size, args.max_vid_label_length).long().to(device)
                    vid_preds = torch.where(mask <= vid_label_lengths, vid_preds, zeros)
                    vid_preds = torch.zeros(batch_size, args.num_classes+1).to(device).scatter(1, vid_preds, 1).long()
                    vid_preds = vid_preds[:, 1:]
                    vid_labels = torch.zeros(batch_size, args.num_classes+1).to(device).scatter(1, vid_labels, 1).long()
                    vid_labels = vid_labels[:, 1:]
                    vid_label_corrects = (vid_labels * vid_preds).sum().float()

                    total_loss = vid_label_loss / vid_label_size
                    if args.use_conv_loss == True:
                        total_loss += conv_loss / batch_size
                    #total_loss = vid_label_loss / vid_label_size + vid_cent_loss / vid_label_size

                    if phase == 'train':
                        total_loss.backward()
                        _ = nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                        optimizer.step()

                running_vid_label_loss += vid_label_loss.item()
                running_vid_label_corrects += vid_label_corrects.item()
                #running_vid_cent_loss += vid_cent_loss.item()
                running_vid_label_size += vid_label_size
                running_num_vid_labels += num_vid_labels.item()
                if args.use_conv_loss == True:
                    running_conv_loss += conv_loss.item()
                    running_conv_size += batch_size
                
                #if args.which_challenge == 'xxx_challenge':
                #    running_time_label_size += time_label_size.item()
                #    running_time_loss += time_loss.item()

            epoch_vid_label_loss = running_vid_label_loss / running_vid_label_size
            #epoch_vid_cent_loss = running_vid_cent_loss / running_vid_label_size
            #epoch_time_loss = 0.0
            epoch_total_loss = epoch_vid_label_loss
            if args.use_conv_loss == True:
                epoch_conv_loss = running_conv_loss / running_conv_size 
                epoch_total_loss += epoch_conv_loss
            #epoch_total_loss = epoch_vid_label_loss + epoch_vid_cent_loss
            epoch_vid_label_recall = running_vid_label_corrects / running_num_vid_labels

            #if args.which_challenge == 'xxx_challenge':
            #    epoch_time_loss = running_time_loss / running_time_label_size
            #    epoch_total_loss = epoch_vid_label_loss + epoch_time_loss

            print('| {} SET | Epoch [{:02d}/{:02d}]'.format(phase.upper(), epoch+1, args.num_epochs))
            print('\t*- Total Loss        : {:.4f}'.format(epoch_total_loss))
            print('\t*- Video Label Loss  : {:.4f}'.format(epoch_vid_label_loss))
            if args.use_conv_loss == True:
                print('\t*- Conv Loss         : {:.4f}'.format(epoch_conv_loss))
            print('\t*- Video Label Recall: {:.4f}'.format(epoch_vid_label_recall))

            # Log the loss in an epoch.
            with open(os.path.join(os.getcwd(), 'logs/{}-log-epoch-{:02}.txt').format(phase, epoch+1), 'w') as f:
                f.write(str(epoch+1) + '\t' +
                        str(epoch_total_loss) + '\t' +
                        str(epoch_vid_label_loss) + '\t' +
                        str(epoch_vid_label_recall))

            # Save the model check points.
            if phase == 'train' and (epoch+1) % args.save_step == 0:
                torch.save({'epoch': epoch+1,
                            'state_dict': model.state_dict()},
                           os.path.join(os.getcwd(), 'models/model-epoch-{:02d}.ckpt'.format(epoch+1)))
            time_elapsed = time.time() - since
            print('=> Running time in a epoch: {:.0f}h {:.0f}m {:.0f}s'
                  .format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))
        scheduler.step()
        print()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str,
                        default='/run/media/hoosiki/WareHouse1/mtb/datasets/VU/pytorch_datasets/',
                        help='input directory for video understanding challenge.')

    parser.add_argument('--which_challenge', type=str, default='2nd_challenge',
                        help='(2nd_challenge) / (3rd_challenge).')

    parser.add_argument('--model_name', type=str, default='transformer',
                        help='transformer, base.')

    parser.add_argument('--use_conv_loss', type=bool, default=True,
                        help='use conv loss but it has not large effect.')

    parser.add_argument('--load_model', type=bool, default=False,
                        help='load_model.')

    parser.add_argument('--max_frame_length', type=int, default=300,
                        help='the maximum length of frame. (301)')

    parser.add_argument('--max_seg_length', type=int, default=60,
                        help='the maximum length of segment step. (60)')

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
                        help='the number of classes. (1000) / (3862)')

    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout. (0.1)')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for training. (0.01)')

    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping. (0.25)')

    parser.add_argument('--step_size', type=int, default=7,
                        help='period of learning rate decay. (5)')

    parser.add_argument('--gamma', type=float, default=0.5,
                        help='multiplicative factor of learning rate decay. (0.1)')

    parser.add_argument('--focal_loss_gamma', type=int, default=0,
                        help='gamma of focal loss. (5)')

    parser.add_argument('--lambda_factor', type=float, default=10.,
                        help='multiplicative factor of segment loss. (0.1)')

    parser.add_argument('--num_epochs', type=int, default=100,
                        help='the number of epochs. (100)')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size. (64) / (256)')

    parser.add_argument('--num_workers', type=int, default=16,
                        help='the number of processes working on cpu. (16)')

    parser.add_argument('--save_step', type=int, default=1,
                        help='save step of model. (1)')

    args = parser.parse_args()

    main(args)
