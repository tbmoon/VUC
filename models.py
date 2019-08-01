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

    def __init__(self, num_seg_frames, rgb_feature_size, audio_feature_size, num_layers, hidden_size, fc_size, num_classes):
        super(GlobalGruModel, self).__init__()
        self.gru = nn.GRU(rgb_feature_size + audio_feature_size, hidden_size, num_layers)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(num_layers * hidden_size, fc_size)
        self.fc2 = nn.Linear(fc_size, num_classes)

    def forward(self, input_seg, input_seg_len):
        '''
        output = self.gru(input)
          input:
          - input_seg: [num_frames, batch_size, feature_size]
          output:
          - hidden: [num_layers, batch_size, hidden_size]
        '''
        packed = pack_padded_sequence(input_seg.float(), input_seg_len, enforce_sorted=False) 
        _, hidden = self.gru(packed)

        hidden = hidden.transpose(0, 1)                # hidden: [batch_size, num_layers, hidden_size]
        output = self.tanh(hidden)                     # output: [batch_size, num_layers, hidden_size]
        output = output.reshape(output.size()[0], -1)  # output: [batch_size, num_layers * hidden_size]
        output = self.fc1(output)                      # output: [batch_size, fc_size]
        output = self.tanh(output)
        output = self.fc2(output)                      # output: [batch_size, num_classes]

        return output
