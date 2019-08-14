import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LstmModel(nn.Module):

    def __init__(self, num_seg_frames, rgb_feature_size, audio_feature_size, num_layers, hidden_size, fc_size, num_classes):
        super(LstmModel, self).__init__()
        self.lstm = nn.LSTM(rgb_feature_size + audio_feature_size, hidden_size, num_layers)
        self.maxpool1d = nn.MaxPool1d(num_seg_frames, stride=1)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.fc2 = nn.Linear(fc_size, num_classes)

    def forward(self, input_seg, last_hidden, last_cell):
        '''
        output = self.lstm(input)
          input:
          - input_seg:   [num_seg_frames, batch_size, feature_size]
          - last_hidden: [1, batch_size, hidden_size]
          - last_cell:   [1, batch_size, hidden_size]
          output:
          - output:      [num_seg_frames, batch_size, hidden_size]
          - last_hidden: [1, batch_size, hidden_size]
          - last_ceel:   [1, batch_size, hidden_size]
        '''
        input_seg = self.tanh(input_seg)
        output, (last_hidden, last_cell) = self.lstm(input_seg.float(), (last_hidden, last_cell))

        output = output.transpose(0, 2)  # output: [hidden_size, batch_size, num_seg_frames]
        output = self.maxpool1d(output)  # output: [hidden_size, batch_size, 1]
        output = output.transpose(0, 2)  # output: [1, batch_size, hidden_size]

        output = self.fc1(output)        # output: [1, batch_size, fc_size]
        output = self.tanh(output)
        output = self.fc2(output)        # output: [1, batch_size, num_classes = 1001]

        return output, (last_hidden, last_cell)


class GlobalGruModel(nn.Module):

    def __init__(self, num_seg_frames, rgb_feature_size, audio_feature_size, embed_size, num_layers, hidden_size, fc_size, num_classes):
        super(GlobalGruModel, self).__init__()
        self.embed = nn.Linear(rgb_feature_size + audio_feature_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers)
        self.tanh = nn.Tanh()
        self.maxpool1d = nn.MaxPool1d(300, stride=1)
        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.fc2 = nn.Linear(fc_size, num_classes)

    def forward(self, padded_frame_rgbs, padded_frame_audios):
        '''
        outputs = self.gru(inputs)
          inputs:
          - padded_frame_rgbs: [batch_size, frame_length, rgb_feature_size]
          - padded_frame_audios: [batch_size, frame_length, audio_feature_size]
          outputs:
          - hidden: [num_layers, batch_size, hidden_size]
        '''
        # padded_frames: [batch_size, frame_length, feature_size]
        padded_frame_rgbs = padded_frame_rgbs / 255.
        padded_frame_audios = padded_frame_audios / 255.
        padded_frames = torch.cat((padded_frame_rgbs, padded_frame_audios), 2)  

        embedded = self.embed(padded_frames)                   # embedded: [batch_size, frame_length, embed_size]
        embedded = self.tanh(embedded)
        embedded = embedded.transpose(0, 1).float()            # embedded: [frame_length, batch_size, feature_size]

        outputs, _ = self.gru(embedded)                        # outputs: [frame_length, batch_size, hidden_size]

        outputs = outputs.transpose(0, 2)                      # outputs: [hidden_size, batch_size, frame_length]
        outputs = self.maxpool1d(outputs)
        outputs = outputs.transpose(0, 2)                      # outputs: [1, batch_size, hidden_size]
        outputs = outputs.squeeze(0)                           # outputs: [batch_size, hidden_size]
        outputs = self.tanh(outputs)

        outputs = self.fc1(outputs)                            # outputs: [batch_size, fc_size]
        outputs = self.tanh(outputs)
        outputs = self.fc2(outputs)                            # outputs: [batch_size, num_classes]

        return outputs
