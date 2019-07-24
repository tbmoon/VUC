import torch
import torch.nn as nn


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
        output, (last_hidden, last_cell) = self.lstm(input_seg.float(), (last_hidden, last_cell))

        output = output.transpose(0, 2)  # output: [hidden_size, batch_size, num_seg_frames]
        output = self.maxpool1d(output)  # output: [hidden_size, batch_size, 1]
        output = output.transpose(0, 2)  # output: [1, batch_size, hidden_size]

        output = self.fc1(output)        # output: [1, batch_size, fc_size]
        output = self.tanh(output)
        output = self.fc2(output)        # output: [1, batch_size, num_classes = 1001]

        return output, (last_hidden, last_cell)
