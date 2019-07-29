import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader import YouTubeDataset, get_dataloader
from models import LstmModel


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(args):

    data_loaders, dataset_sizes = get_dataloader(
        input_dir=args.input_dir,
        phases=['test'],
        num_seg_frames=args.num_seg_frames,
        max_frame_length=args.max_frame_length,
        rgb_feature_size=args.rgb_feature_size,
        audio_feature_size=args.audio_feature_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    checkpoint = torch.load(args.model_dir + args.model_name)
    
    model = LstmModel(
        num_seg_frames=args.num_seg_frames,
        rgb_feature_size=args.rgb_feature_size,
        audio_feature_size=args.audio_feature_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        fc_size=args.fc_size,
        num_classes=args.num_classes).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    with torch.no_grad():
        for batch_idx, batch_sample in enumerate(data_loaders['test']):
            frame_length = batch_sample['frame_length']
            frame_rgb = batch_sample['frame_rgb'].to(device)
            frame_audio = batch_sample['frame_audio'].to(device) 
            
            frame_feature = torch.cat((frame_rgb, frame_audio), 2)  # [batch_size, frame_length, feature_size]
            frame_feature = frame_feature.transpose(0, 1)           # [frame_legnth, batch_size, feature_size]
            last_hidden = torch.zeros(args.num_layers, args.batch_size, args.hidden_size).to(device)
            last_cell = torch.zeros(args.num_layers, args.batch_size, args.hidden_size).to(device)
                    
            preds = []
            probs = []
            for iseg in range(max(frame_length) // args.num_seg_frames):
                output, (last_hidden, last_cell) = model(
                    frame_feature[args.num_seg_frames*iseg:args.num_seg_frames*(iseg+1)],
                    last_hidden,
                    last_cell)                # output: [1, batch_size, num_classes = 1001]
                output = output.squeeze(0)    # output: [batch_size, num_classes = 1001]
                print(output)
                #_, pred = torch.max(output, 1)
                #_, pred = torch.max(output, 0)


                #pred = pred.detach().cpu().numpy()


                #pred = pred.detach().cpu().numpy()[0]
                #prob = F.softmax(output, dim=1).detach().cpu().numpy()[0]
                #if (pred != 0):
                #    print(pred)
            
                #preds.append(pred)
                #probs.append(prob)
            
            #print(preds)
            #print(probs)
            #print()
            if batch_idx == 2:
                break




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str,
                        default='/run/media/hoosiki/WareHouse3/mtb/datasets/VU/active_datasets/',
                        help='input directory for video understanding challenge.')

    parser.add_argument('--model_dir', type=str, default='./models/',
                        help='directory for saved models.')

    parser.add_argument('--model_name', type=str, default='model-epoch-01.ckpt',
                        help='name of model.')

    parser.add_argument('--num_seg_frames', type=int, default=5,
                        help='the number of frames per segment.')
    
    parser.add_argument('--max_frame_length', type=int, default=300,
                        help='maximum length of frame. \
                              the length in the VQA dataset = 300.')
    
    parser.add_argument('--rgb_feature_size', type=int, default=1024,
                        help='rgb feature size in a frame.')

    parser.add_argument('--audio_feature_size', type=int, default=128,
                        help='audio feature size in a frame.')

    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of layers of the RNN(LSTM).')
    
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='hidden_size in the LSTM.')
    
    parser.add_argument('--fc_size', type=int, default=1024,
                        help='the number of classes.')
    
    parser.add_argument('--num_classes', type=int, default=1001,
                        help='the number of classes.')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size.')

    parser.add_argument('--num_workers', type=int, default=1,
                        help='number of processes working on cpu.')

    args = parser.parse_args()

    main(args)
