import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import lr_scheduler
from data_loader import YouTubeDataset, get_dataloader
from models import LstmModel, GlobalGruModel


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(args):

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    data_loaders, dataset_sizes = get_dataloader(
        input_dir=args.input_dir,
        phases=['train', 'valid'],
        num_seg_frames=args.num_seg_frames,
        max_frame_length=args.max_frame_length,
        rgb_feature_size=args.rgb_feature_size,
        audio_feature_size=args.audio_feature_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    model = GlobalGruModel(
        num_seg_frames=args.num_seg_frames,
        rgb_feature_size=args.rgb_feature_size,
        audio_feature_size=args.audio_feature_size,
        embed_size=args.embed_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        fc_size=args.fc_size,
        num_classes=args.num_classes).to(device)

    #checkpoint = torch.load(args.model_dir + '/model-epoch-01.ckpt')
    #model.load_state_dict(checkpoint['state_dict'])
    
    criterion = nn.CrossEntropyLoss()

    params = model.parameters()

    optimizer = optim.Adam(params, lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    for epoch in range(args.num_epochs):
        for phase in ['train']:
            running_loss = 0.0
            running_corrects = 0

            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            for idx, (padded_frame_rgbs, padded_frame_audios, frame_lengths, video_labels) in enumerate(data_loaders[phase]):
                optimizer.zero_grad()

                # padded_frame_rgbs: [batch_size, frame_lengths, rgb_feature_size]
                padded_frame_rgbs = padded_frame_rgbs.to(device)
                padded_frame_audios = padded_frame_audios.to(device)
                frame_lengths = frame_lengths
                video_labels = video_labels.to(device)
                hidden = torch.zeros(args.num_layers, args.batch_size, args.hidden_size).to(device)
                
                # padded_frame_rgbs: [frame_lengths, batch_size, rgb_feature_size]
                padded_frame_rgbs = padded_frame_rgbs.transpose(0, 1)
                padded_frame_audios = padded_frame_audios.transpose(0, 1)

                with torch.set_grad_enabled(phase == 'train'):
                    loss = 0.0
                    
                    # outputs: [batch_size, num_classes = 1001]
                    for iframe in range(max(frame_lengths)):
                        outputs, hidden = model(padded_frame_rgbs[iframe], padded_frame_audios[iframe], frame_lengths, hidden)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, video_labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * padded_frame_rgbs.size(0)
                running_corrects += torch.sum(preds == video_labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

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
        print()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str,
                        default='/run/media/hoosiki/WareHouse3/mtb/datasets/VU/active_datasets/',
                        help='input directory for video understanding challenge.')

    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='directory for logs.')

    parser.add_argument('--model_dir', type=str, default='./models',
                        help='directory for saved models.')

    parser.add_argument('--num_seg_frames', type=int, default=5,
                        help='the number of frames per segment.')
    
    parser.add_argument('--max_frame_length', type=int, default=300,
                        help='maximum length of frame. \
                              the length in the VQA dataset = 300.')
    
    parser.add_argument('--rgb_feature_size', type=int, default=1024,
                        help='rgb feature size in a frame.')

    parser.add_argument('--audio_feature_size', type=int, default=128,
                        help='audio feature size in a frame.')

    parser.add_argument('--embed_size', type=int, default=128,
                        help='embedding size.')

    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of layers of the RNN(LSTM).')
    
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='hidden_size in the LSTM.')
    
    parser.add_argument('--fc_size', type=int, default=256,
                        help='the number of classes.')
    
    parser.add_argument('--num_classes', type=int, default=1001,
                        help='the number of classes.')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for training.')

    parser.add_argument('--step_size', type=int, default=20,
                        help='period of learning rate decay.')

    parser.add_argument('--gamma', type=float, default=0.1,
                        help='multiplicative factor of learning rate decay.')

    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs.')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size.')

    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of processes working on cpu.')

    parser.add_argument('--save_step', type=int, default=1,
                        help='save step of model.')

    args = parser.parse_args()

    main(args)
