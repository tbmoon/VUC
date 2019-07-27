import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from data_loader import YouTubeDataset, get_dataloader
from models import LstmModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    data_loader = get_dataloader(
        input_dir=args.input_dir,
        max_frame_length=args.max_frame_length,
        rgb_feature_size=args.rgb_feature_size,
        audio_feature_size=args.audio_feature_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    model = LstmModel(
        num_seg_frames=args.num_seg_frames,
        rgb_feature_size=args.rgb_feature_size,
        audio_feature_size=args.audio_feature_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        fc_size=args.fc_size,
        num_classes=args.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()

    params = model.parameters()

    optimizer = optim.Adam(params, lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    for epoch in range(args.num_epochs):

        #for phase in ['train', 'valid']:
        for phase in ['train']: 

            running_loss = 0.0

            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            for batch_idx, batch_sample in enumerate(data_loader[phase]):
                optimizer.zero_grad()
                
                frame_length = batch_sample['frame_length']
                frame_rgb = batch_sample['frame_rgb'].to(device)
                frame_audio = batch_sample['frame_audio'].to(device) 
                segment_labels = batch_sample['segment_labels']
                segment_scores = batch_sample['segment_scores']
                segment_start_times = batch_sample['segment_start_times']

                frame_feature = torch.cat((frame_rgb, frame_audio), 2)  # [batch_size, frame_length, feature_size]
                frame_feature = frame_feature.transpose(0, 1)           # [frame_legnth, batch_size, feature_size]
                last_hidden = torch.randn(args.num_layers, args.batch_size, args.hidden_size).to(device)
                last_cell = torch.randn(args.num_layers, args.batch_size, args.hidden_size).to(device)
                
                with torch.set_grad_enabled(phase == 'train'):
                    
                    for iseg in range(max(frame_length) // args.num_seg_frames):
                        
                        output, (last_hidden, last_cell) = model(
                            frame_feature[args.num_seg_frames*iseg:args.num_seg_frames*(iseg+1)],
                            last_hidden,
                            last_cell)  # output: [1, batch_size, num_classes = 1001]
                        
                        
                        ###########
                        '''
                        라벨에 대해서 전처리하자!!
                        미니배치 샘플은 각각 다른 라벨 크기를 가지고 있어서 문제가 생김.
                        따라서, 0으로 패딩 후 각각의 스타트 타임에 라벨을 리스트에 주고 텐서로 변화하자.
                        '''
                        #print(output)
                        #print(output.shape)


                        
                        print(segment_labels)
                        print(segment_labels.numpy())
                        print(segment_labels.shape)
                        print(segment_start_times)
                        print(segment_start_times.shape)
                        #loss = criterion(output, segment_labels)
                        #print(output)
                        break
                    
                    #loss = criterion(output, label)
                    
                    if phase == 'train':
                        #loss.backward()
                        optimizer.step()

                #print(frame_rgb)
                #print(batch_idx)
                break
                    


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
    
    
    parser.add_argument('--max_frame_length', type=int, default=301,
                        help='maximum length of frame. \
                              the length in the VQA dataset = 301')
    
    parser.add_argument('--rgb_feature_size', type=int, default=1024,
                        help='rgb feature size in a frame.')

    parser.add_argument('--audio_feature_size', type=int, default=128,
                        help='audio feature size in a frame.')

    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers of the RNN(LSTM).')
    
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='hidden_size in the LSTM.')
    
    parser.add_argument('--fc_size', type=int, default=1024,
                        help='the number of classes.')
    
    parser.add_argument('--num_classes', type=int, default=1001,
                        help='the number of classes.')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for training.')

    parser.add_argument('--step_size', type=int, default=10,
                        help='period of learning rate decay.')

    parser.add_argument('--gamma', type=float, default=0.1,
                        help='multiplicative factor of learning rate decay.')

    parser.add_argument('--num_epochs', type=int, default=1,
                        help='number of epochs.')

    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch_size.')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of processes working on cpu.')

    parser.add_argument('--save_step', type=int, default=1,
                        help='save step of model.')

    args = parser.parse_args()

    main(args)