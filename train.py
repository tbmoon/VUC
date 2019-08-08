import os
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from data_loader import YouTubeDataset, get_dataloader
from models import TransformerModel


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
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    model = TransformerModel(
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        rgb_feature_size=args.rgb_feature_size,
        audio_feature_size=args.audio_feature_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_classes=args.num_classes,
        dropout=args.dropout)
    
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    #checkpoint = torch.load(args.model_dir + '/model-epoch-01.ckpt')
    #model.load_state_dict(checkpoint['state_dict'])
    
    criterion = nn.CrossEntropyLoss().to(device)

    params = model.parameters()

    optimizer = optim.Adam(params, lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    for epoch in range(args.num_epochs):
        since = time.time()
        for phase in ['train', 'valid']:
            running_loss = 0.0
            running_corrects = 0

            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            for idx, (padded_frame_rgbs, padded_frame_audios, video_labels) in enumerate(data_loaders[phase]):
                optimizer.zero_grad()

                # padded_frame_rgbs: [batch_size, max_frame_length=301, rgb_feature_size]
                padded_frame_rgbs = padded_frame_rgbs.to(device)
                padded_frame_audios = padded_frame_audios.to(device)
                # a label per sample: it shoulde be updated for multi-lable classification later.
                video_labels = video_labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    loss = 0.0

                    # outputs: [batch_size, num_classes = 1001]
                    outputs = model(padded_frame_rgbs, padded_frame_audios)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, video_labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * padded_frame_rgbs.size(0)
                running_corrects += torch.sum(preds == video_labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects.item()) / dataset_sizes[phase]

            print('| {} SET | Epoch [{:02d}/{:02d}], Loss: {:.4f}, Acc: {:.4f}'
                  .format(phase.upper(), epoch+1, args.num_epochs, epoch_loss, epoch_acc))

            # Log the loss in an epoch.
            with open(os.path.join(args.log_dir, '{}-log-epoch-{:02}.txt')
                      .format(phase, epoch+1), 'w') as f:
                f.write(str(epoch+1) + '\t'
                        + str(epoch_loss) + '\t'
                        + str(epoch_acc))

        # Save the model check points.
        if (epoch+1) % args.save_step == 0:
            torch.save({'epoch': epoch+1, 'state_dict': model.state_dict()},
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
