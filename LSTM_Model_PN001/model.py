import torch
import torch.nn as nn

class SeizureLSTM(nn.Module):
    def __init__(self, input_size=18, hidden_size=64, num_layers=1):
        super(SeizureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch, seq, hidden]
        last_out = lstm_out[:, -1, :]  # get output from last timestep
        out = self.fc(last_out)
        return self.sigmoid(out)
