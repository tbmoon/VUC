import torch
import torch.nn as nn


class LstmModel(nn.Module):

    def __init__(self, rgb_feature_size, audio_feature_size, num_layers, hidden_size, fc_size, num_classes):
        super(LstmModel, self).__init__()
        self.lstm = nn.LSTM(rgb_feature_size + audio_feature_size, hidden_size, num_layers)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.fc2 = nn.Linear(fc_size, num_classes)

    def forward(self, input_step, last_hidden):
        output, (hidden, cell) = self.lstm(input_step, last_hidden)
        output = self.fc1(output)
        output = self.tanh(output)
        output = self.fc2(output)

        return outputs, hidden
