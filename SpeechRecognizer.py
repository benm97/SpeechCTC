import torch.nn as nn
import torch.nn.functional as F


class SpeechRecognitionModel(nn.Module):
    def __init__(self, input_dim, output_dim, activation='leaky_relu', dropout=0.2):
        super(SpeechRecognitionModel, self).__init__()

        self.activation = activation

        # Convolution layer 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Convolution layer 2
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        # RNN layers
        self.rnn1 = nn.LSTM(928, 128, bidirectional=True,
                            batch_first=True, dropout=dropout)

        self.rnn2 = nn.LSTM(256, 128, bidirectional=True, batch_first=True, dropout=dropout)

        # self.rnn3 = nn.LSTM(256, 128, bidirectional=True, batch_first=True, dropout=dropout)
        #
        # self.rnn4 = nn.LSTM(256, 128, bidirectional=True, batch_first=True, dropout=dropout)
        #
        # self.rnn5 = nn.LSTM(256, 128, bidirectional=True, batch_first=True)

        # Dense layer
        self.fc1 = nn.Linear(256, 256)

        # Classification layer
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        # expand dims to add channel dimension
        x = x.unsqueeze(1)

        # Convolution layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activate(x)

        # Convolution layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activate(x)

        # Reshape the resulted volume to feed the RNNs layers
        B, C, H, W = x.size()
        x = x.view(B, H, C * W)

        # RNN layers
        x, _ = self.rnn1(x)

        x, _ = self.rnn2(x)

        # x, _ = self.rnn3(x)
        #
        # x, _ = self.rnn4(x)
        #
        # x, _ = self.rnn5(x)

        # Dense layer
        x = self.fc1(x)
        x = self.activate(x)

        # Classification layer
        x = self.fc2(x)
        x = F.log_softmax(x, dim=-1)

        return x

    def activate(self, x):
        if self.activation == 'leaky_relu':
            return F.leaky_relu(x)
        # Add other activation functions if needed
        return x
