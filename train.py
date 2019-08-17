import os
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from data_loader import YouTubeDataset, get_dataloader
from models import TransformerEncoder, RNNDecoder


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(args):

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    data_loaders, dataset_sizes = get_dataloader(
        input_dir=args.input_dir,
        phases=['train', 'valid'],
        max_frame_length=args.max_frame_length,
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
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout)

    decoder = RNNDecoder(
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_classes=args.num_classes,
        dropout=args.dropout)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in encoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if args.load_model == True:
        checkpoint = torch.load(args.model_dir + '/model-epoch-10.ckpt')
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

    criterion = nn.CrossEntropyLoss().to(device)

    encoder_params = encoder.parameters()
    decoder_params = decoder.parameters()

    encoder_optimizer = optim.Adam(encoder_params, lr=args.learning_rate)
    decoder_optimizer = optim.Adam(decoder_params, lr=args.learning_rate)

    encoder_scheduler = lr_scheduler.StepLR(encoder_optimizer, step_size=args.step_size, gamma=args.gamma)
    decoder_scheduler = lr_scheduler.StepLR(decoder_optimizer, step_size=args.step_size, gamma=args.gamma)

    for epoch in range(args.num_epochs):
        since = time.time()
        for phase in ['train', 'valid']:
            running_loss = 0.0
            running_corrects = 0

            if phase == 'train':
                encoder_scheduler.step()
                decoder_scheduler.step()
                encoder.train()
                decoder.train()
            else:
                encoder.eval()
                decoder.eval()

            for idx, (padded_frame_rgbs, padded_frame_audios, video_labels) in enumerate(data_loaders[phase]):
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                # padded_frame_rgbs: [batch_size, max_frame_length=300, rgb_feature_size=1024]
                # padded_frame_audios: [batch_size, max_frame_length=300, audio_feature_size=128]
                # vidoe_labels: [batch_size]
                padded_frame_rgbs = padded_frame_rgbs.to(device)
                padded_frame_audios = padded_frame_audios.to(device)
                video_labels = video_labels.to(device)
                batch_size = video_labels.size(0)

                with torch.set_grad_enabled(phase == 'train'):
                    loss = 0.0

                    # seq_features: [batch_size, seq_length=60, d_model]
                    # decoder_input: [1, batch_size, d_model]
                    # decoder_hidden: [1, batch_size, d_model]
                    seq_features = encoder(padded_frame_rgbs, padded_frame_audios)
                    decoder_input, decoder_hidden = decoder.init_input_hidden(batch_size, device)
                    
                    for itarget in range(args.max_target_length):
                        raw_attn_weights, decoder_input, decoder_hidden, output = \
                            decoder(decoder_input, decoder_hidden, seq_features)

                        _, pred = torch.max(output, 1)
                        
                        loss += criterion(output, video_labels)

                    if phase == 'train':
                        loss.backward()
                        encoder_optimizer.step()
                        decoder_optimizer.step()

                # one classe will be predicted. this should be updated.
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(pred == video_labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects.item()) / dataset_sizes[phase]

            print('| {} SET | Epoch [{:02d}/{:02d}], Loss: {:.4f}, Acc: {:.4f}'
                  .format(phase.upper(), epoch+1, args.num_epochs, epoch_loss, epoch_acc))

            # Log the loss in an epoch.
            with open(os.path.join(args.log_dir, '{}-log-epoch-{:02}.txt').format(phase, epoch+1), 'w') as f:
                f.write(str(epoch+1) + '\t' + str(epoch_loss) + '\t' + str(epoch_acc))

        # Save the model check points.
        if (epoch+1) % args.save_step == 0:
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
                        default='/run/media/hoosiki/WareHouse3/mtb/datasets/VU/pickled_datasets/2nd_challenge/',
                        help='input directory for video understanding challenge.')

    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='directory for logs.')

    parser.add_argument('--model_dir', type=str, default='./models',
                        help='directory for saved models.')
    
    parser.add_argument('--load_model', type=bool, default=True,
                        help='load_model.')

    parser.add_argument('--max_frame_length', type=int, default=300,
                        help='the maximum length of frame. (301)')

    parser.add_argument('--max_target_length', type=int, default=1,
                        help='the maximum length of target. (?)')

    parser.add_argument('--n_layers', type=int, default=6,
                        help='n_layers for the encoder. (6)')

    parser.add_argument('--n_heads', type=int, default=8,
                        help='n_heads for the attention. (8)')

    parser.add_argument('--rgb_feature_size', type=int, default=1024,
                        help='rgb feature size in a frame. (1024)')

    parser.add_argument('--audio_feature_size', type=int, default=128,
                        help='audio feature size in a frame. (128)')

    parser.add_argument('--d_model', type=int, default=64,
                        help='d_model for feature projection. (64)')

    parser.add_argument('--d_ff', type=int, default=128,
                        help='d_ff. (128)')

    parser.add_argument('--num_classes', type=int, default=1001,
                        help='the number of classes. (1000+1) / (3862)')

    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout. (0.1)')

    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate for training. (0.01)')

    parser.add_argument('--step_size', type=int, default=10,
                        help='period of learning rate decay. (10)')

    parser.add_argument('--gamma', type=float, default=0.1,
                        help='multiplicative factor of learning rate decay. (0.1)')

    parser.add_argument('--num_epochs', type=int, default=100,
                        help='the number of epochs. (100)')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size. (256)')

    parser.add_argument('--num_workers', type=int, default=8,
                        help='the number of processes working on cpu. (16)')

    parser.add_argument('--save_step', type=int, default=1,
                        help='save step of model. (1)')

    args = parser.parse_args()

    main(args)
