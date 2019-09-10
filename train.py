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
from models import TransformerEncoder, RNNDecoder


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def cross_entropy_loss_with_vid_label_processing(logit, labels):
    '''
    inputs:
        - logit: [batch_size, num_classes - 1]
        - labels: [batch_size, max_vid_label_length]
    outputs:
        - loss: [1]
        - labels: [batch_size, max_vid_label_length]
        - selected_label: [batch_size]
    '''
    eps = 1e-6
    batch_size = logit.size(0)
    max_vid_label_length = labels.size(1)

    # Do NOT include 0-label prediction in softmax calculation.
    prob = F.softmax(logit, dim=1)
    zero_col = torch.zeros(batch_size, 1).to(device)
    prob = torch.cat((zero_col, prob), dim=1)

    # Probability of predicted labels.
    prob_candidates = torch.gather(prob, 1, labels)
    selected_prob, selected_label_idx = torch.max(prob_candidates, dim=1)
    selected_label = labels[range(batch_size), selected_label_idx]
    mask = selected_label.float().ge(0.5)

    zeros = torch.zeros(batch_size, max_vid_label_length).to(device)
    labels = torch.where(labels == selected_label.view(-1, 1), zeros, labels.float())
    labels = labels.long()

    loss = -1 * (1-selected_prob)**args.focal_loss_gamma * torch.log(selected_prob + eps)

    #if args.which_challenge == '2nd_challenge':
    #    batch_prob = torch.rand(batch_size).to(device)
    #    threshold = np.array([0. for p in range(args.num_classes)])
    #    threshold[0] = 1.1
    #    threshold[1] = 0.95
    #    threshold[2:5] = 0.90
    #    threshold[5:50] = 0.80
    #    threshold[50:100] = 0.30
    #    threshold = torch.tensor(threshold, dtype=torch.float).to(device)
    #    batch_threshold = torch.gather(threshold, 0, selected_label)
    #    zeros = torch.zeros(batch_size).byte().to(device)
    #    mask = torch.where(batch_prob > batch_threshold, mask, zeros)

    loss = loss.masked_select(mask).sum()

    return loss, labels, selected_label


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
    args.num_vid_label_pred = 18 if args.which_challenge == '2nd_challenge' else 4

    data_loaders, dataset_sizes = get_dataloader(
        input_dir=args.input_dir,
        which_challenge=args.which_challenge,
        phases=['train', 'valid'],
        max_frame_length=args.max_frame_length,
        max_vid_label_length=args.max_vid_label_length,
        max_seg_label_length=args.max_seg_label_length,
        rgb_feature_size=args.rgb_feature_size,
        audio_feature_size=args.audio_feature_size,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    encoder = TransformerEncoder(
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        rgb_feature_size=args.rgb_feature_size,
        audio_feature_size=args.audio_feature_size,
        d_rgb=args.d_rgb,
        d_audio=args.d_audio,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout)

    decoder = RNNDecoder(
        d_model=args.d_model,
        d_linear=args.d_linear,
        num_classes=args.num_classes,
        dropout=args.dropout)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    center_loss = CenterLoss(num_classes=args.num_classes, feat_dim=args.d_model, device=device)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in encoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if args.load_model == True:
        checkpoint = torch.load(args.model_dir + '/model-epoch-pretrained.ckpt')
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder_params = encoder.parameters()
    decoder_params = list(center_loss.parameters()) + list(decoder.parameters())

    encoder_optimizer = optim.Adam(encoder_params, lr=args.learning_rate)
    decoder_optimizer = optim.Adam(decoder_params, lr=args.learning_rate)

    encoder_scheduler = lr_scheduler.StepLR(encoder_optimizer, step_size=args.step_size, gamma=args.gamma)
    decoder_scheduler = lr_scheduler.StepLR(decoder_optimizer, step_size=args.step_size, gamma=args.gamma)

    for epoch in range(args.num_epochs):
        since = time.time()
        for phase in ['train', 'valid']:
            running_vid_loss = 0.0
            running_vid_cent_loss = 0.0
            running_time_loss = 0.0
            running_vid_label_size = 0
            running_time_label_size = 0
            running_vid_corrects = 0

            if phase == 'train':
                encoder_scheduler.step()
                decoder_scheduler.step()
                encoder.train()
                decoder.train()
            else:
                encoder.eval()
                decoder.eval()

            for idx, (frame_lengths, frame_rgbs, frame_audios, vid_labels, seg_labels, seg_times) \
                in enumerate(data_loaders[phase]):
                
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

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
                vid_label_size = vid_labels.float().ge(0.5).sum()
                time_label_size = 0

                with torch.set_grad_enabled(phase == 'train'):
                    total_loss = 0.0
                    vid_loss = 0.0
                    vid_cent_loss = 0.0
                    time_loss = 0.0
                    vid_corrects = 0
                    time_corrects = 0

                    # seq_features: [batch_size, seq_length=60, d_model]
                    # decoder_input: [1, batch_size, d_model]
                    # decoder_hidden: [1, batch_size, d_model]
                    seq_features = encoder(frame_rgbs, frame_audios)
                    decoder_input, decoder_hidden = decoder.init_input_hidden(batch_size, device)

                    for itarget in range(args.num_vid_label_pred):
                        raw_attn_weights, decoder_input_next, decoder_hidden, vid_logit = \
                            decoder(decoder_input, decoder_hidden, seq_features)

                        # Do NOT include 0-label in softmax calculation and prediction.
                        vid_logit = vid_logit[:, 1:]
                        _, vid_pred = torch.max(vid_logit, dim=1)
                        vid_pred = vid_pred + 1
                        v_loss, vid_labels, selected_vid_label = \
                            cross_entropy_loss_with_vid_label_processing(vid_logit, vid_labels)
                        
                        if args.which_challenge == 'xxx_challenge':
                            _, time_pred = torch.max(raw_attn_weights, dim=1)
                            time_pred = time_pred + 1
                            t_loss, t_label_size = \
                                binary_cross_entropy_loss_with_seg_label_processing(frame_lengths,
                                                                                    selected_vid_label,
                                                                                    raw_attn_weights,
                                                                                    seg_labels,
                                                                                    seg_times)
                        vid_loss += v_loss
                        vid_cent_loss += center_loss(decoder_input.squeeze(0), selected_vid_label, device)
                        
                        decoder_input = decoder_input_next
                        zeros = torch.zeros(batch_size, dtype=torch.long).to(device)
                        mask = 1 - torch.eq(selected_vid_label, zeros)
                        vid_correct = torch.eq(selected_vid_label, vid_pred)
                        vid_correct = vid_correct.masked_select(mask)
                        vid_corrects += torch.sum(vid_correct)

                        if args.which_challenge == 'xxx_challenge':
                            time_loss += args.lambda_factor * t_loss
                            time_label_size += t_label_size
                            total_loss = vid_loss / vid_label_size + time_loss / time_label_size

                    total_loss = vid_loss / vid_label_size + vid_cent_loss / vid_label_size

                    if phase == 'train':
                        total_loss.backward()
                        _ = nn.utils.clip_grad_norm_(encoder.parameters(), args.clip)
                        _ = nn.utils.clip_grad_norm_(decoder.parameters(), args.clip)
                        encoder_optimizer.step()
                        decoder_optimizer.step()

                running_vid_loss += vid_loss.item()
                running_vid_cent_loss += vid_cent_loss.item()
                running_vid_label_size += vid_label_size.item()
                running_vid_corrects += vid_corrects.item()
                if args.which_challenge == 'xxx_challenge':
                    running_time_label_size += time_label_size.item()
                    running_time_loss += time_loss.item()

            epoch_vid_loss = running_vid_loss / running_vid_label_size
            epoch_vid_cent_loss = running_vid_cent_loss / running_vid_label_size
            epoch_vid_acc = float(running_vid_corrects) / running_vid_label_size
            epoch_time_loss = 0.0
            epoch_total_loss = epoch_vid_loss + epoch_vid_cent_loss

            if args.which_challenge == 'xxx_challenge':
                epoch_time_loss = running_time_loss / running_time_label_size
                epoch_total_loss = epoch_vid_loss + epoch_time_loss

            print('| {} SET | Epoch [{:02d}/{:02d}], Total Loss: {:.4f}, Video Loss: {:.4f}, Cent Loss: {:.4f}, Video Acc: {:.4f}' \
                  .format(phase.upper(), epoch+1, args.num_epochs, \
                          epoch_total_loss, epoch_vid_loss, epoch_vid_cent_loss, epoch_vid_acc))

            # Log the loss in an epoch.
            with open(os.path.join(args.log_dir, '{}-log-epoch-{:02}.txt').format(phase, epoch+1), 'w') as f:
                f.write(str(epoch+1) + '\t' + str(epoch_vid_loss) + '\t' + str(epoch_vid_acc))

            # Save the model check points.
            if phase == 'train' and (epoch+1) % args.save_step == 0:
                torch.save({'epoch': epoch+1,
                            'encoder_state_dict': encoder.state_dict(),
                            'decoder_state_dict': decoder.state_dict()},
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

    parser.add_argument('--d_ff', type=int, default=256,
                        help='d_ff. 2048 for paper. (1024)')

    parser.add_argument('--d_linear', type=int, default=512,
                        help='d_linear. (2048)')

    parser.add_argument('--num_classes', type=int, default=1001,
                        help='the number of classes. (1000+1) / (3862+1)')

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

    parser.add_argument('--num_epochs', type=int, default=20,
                        help='the number of epochs. (100)')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size. (64) / (256)')

    parser.add_argument('--num_workers', type=int, default=16,
                        help='the number of processes working on cpu. (16)')

    parser.add_argument('--save_step', type=int, default=1,
                        help='save step of model. (1)')

    args = parser.parse_args()

    main(args)
