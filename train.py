import os
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from center_loss import CenterLoss
from data_loader import YouTubeDataset, get_dataloader
from models import TransformerModel


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

    # Use balanced (cross-entropy / focal) loss.
    unselected_weight = is_selected.sum(dim=1).view(-1, 1).float()  # unselected_weight: [batch_size, 1]
    selected_weight = num_classes - unselected_weight               # selected_weight: [batch_size, 1]

    loss = -1 * (1 - probs)**args.focal_loss_gamma * torch.log(probs + eps)
    loss = torch.where(is_selected == 1, selected_weight * loss, unselected_weight * loss)
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

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

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

    model = TransformerModel(
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        rgb_feature_size=args.rgb_feature_size,
        audio_feature_size=args.audio_feature_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        d_proj=args.d_proj,
        num_classes=args.num_classes,
        dropout=args.dropout)

    model = model.to(device)

    center_loss = CenterLoss(num_classes=args.num_classes, feat_dim=args.d_model, device=device)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if args.load_model == True:
        checkpoint = torch.load(args.model_dir + '/model-epoch-pretrained.ckpt')
        model.load_state_dict(checkpoint['model_state_dict'])

    params = list(model.parameters()) + list(center_loss.parameters())
    optimizer = optim.Adam(params, lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    for epoch in range(args.num_epochs):
        since = time.time()
        for phase in ['train', 'valid']:
            running_vid_label_loss = 0.0
            running_vid_cent_loss = 0.0
            running_time_loss = 0.0
            running_vid_label_size = 0
            running_time_label_size = 0
            running_vid_corrects = 0

            if phase == 'train':
                scheduler.step()
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

                with torch.set_grad_enabled(phase == 'train'):
                    total_loss = 0.0
                    vid_label_loss = 0.0
                    vid_cent_loss = 0.0
                    time_loss = 0.0
                    vid_corrects = 0
                    time_corrects = 0

                    # vid_probs: [batch_size, num_classes]
                    vid_probs = model(frame_rgbs, frame_audios, device)

                    vid_label_loss = video_label_loss(vid_probs, vid_labels)
                    total_loss = vid_label_loss / vid_label_size
                    #total_loss = vid_label_loss / vid_label_size + vid_cent_loss / vid_label_size

                                                
                    #    zeros = torch.zeros(batch_size, dtype=torch.long).to(device)
                    #    mask = 1 - torch.eq(selected_vid_label, zeros)
                    #    vid_correct = torch.eq(selected_vid_label, vid_pred)
                    #    vid_correct = vid_correct.masked_select(mask)
                    #    vid_corrects += torch.sum(vid_correct)


                    if phase == 'train':
                        total_loss.backward()
                        _ = nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                        optimizer.step()

                running_vid_label_loss += vid_label_loss.item()
                #running_vid_cent_loss += vid_cent_loss.item()
                running_vid_label_size += vid_label_size
                #if args.which_challenge == 'xxx_challenge':
                #    running_time_label_size += time_label_size.item()
                #    running_time_loss += time_loss.item()
                #break

            epoch_vid_label_loss = running_vid_label_loss / running_vid_label_size
            #epoch_vid_cent_loss = running_vid_cent_loss / running_vid_label_size
            #epoch_time_loss = 0.0
            epoch_total_loss = epoch_vid_label_loss
            #epoch_total_loss = epoch_vid_label_loss + epoch_vid_cent_loss

            #if args.which_challenge == 'xxx_challenge':
            #    epoch_time_loss = running_time_loss / running_time_label_size
            #    epoch_total_loss = epoch_vid_label_loss + epoch_time_loss

            print('| {} SET | Epoch [{:02d}/{:02d}], Total Loss: {:.4f}, Video Label Loss: {:.4f}' \
                  .format(phase.upper(), epoch+1, args.num_epochs, \
                          epoch_total_loss, epoch_vid_label_loss))

            # Log the loss in an epoch.
            with open(os.path.join(args.log_dir, '{}-log-epoch-{:02}.txt').format(phase, epoch+1), 'w') as f:
                f.write(str(epoch+1) + '\t' + str(epoch_total_loss) + '\t' + str(epoch_vid_label_loss))

            # Save the model check points.
            if phase == 'train' and (epoch+1) % args.save_step == 0:
                torch.save({'epoch': epoch+1,
                            'model_state_dict': model.state_dict()},
                           os.path.join(args.model_dir, 'model-epoch-{:02d}.ckpt'.format(epoch+1)))
        time_elapsed = time.time() - since
        print('=> Running time in a epoch: {:.0f}h {:.0f}m {:.0f}s'
              .format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))
        print()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str,
                        default='/run/media/hoosiki/WareHouse2/mtb/datasets/VU/pytorch_datasets/',
                        help='input directory for video understanding challenge.')

    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='directory for logs.')

    parser.add_argument('--model_dir', type=str, default='./models',
                        help='directory for saved models.')

    parser.add_argument('--which_challenge', type=str, default='2nd_challenge',
                        help='(2nd_challenge) / (3rd_challenge).')

    parser.add_argument('--load_model', type=bool, default=True,
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

    parser.add_argument('--d_model', type=int, default=128,
                        help='d_model for feature projection. \
                              512 for paper. (256)')

    parser.add_argument('--d_proj', type=int, default=64,
                        help='d_proj for q, k, v projection. (64)')

    parser.add_argument('--d_ff', type=int, default=256,
                        help='d_ff. 2048 for paper. (1024)')

    parser.add_argument('--d_linear', type=int, default=512,
                        help='d_linear. (2048)')

    parser.add_argument('--num_classes', type=int, default=1000,
                        help='the number of classes. (1000) / (3862)')

    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout. (0.1)')

    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate for training. (0.01)')

    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping. (0.25)')

    parser.add_argument('--step_size', type=int, default=3,
                        help='period of learning rate decay. (10)')

    parser.add_argument('--gamma', type=float, default=0.1,
                        help='multiplicative factor of learning rate decay. (0.1)')

    parser.add_argument('--focal_loss_gamma', type=int, default=0,
                        help='gamma of focal loss. (5)')

    parser.add_argument('--lambda_factor', type=float, default=10.,
                        help='multiplicative factor of segment loss. (0.1)')

    parser.add_argument('--num_epochs', type=int, default=10,
                        help='the number of epochs. (100)')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size. (64) / (256)')

    parser.add_argument('--num_workers', type=int, default=16,
                        help='the number of processes working on cpu. (16)')

    parser.add_argument('--save_step', type=int, default=1,
                        help='save step of model. (1)')

    args = parser.parse_args()

    main(args)
