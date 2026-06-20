import torch
import torch.nn as nn

class TrafficCNNLSTMModel(nn.Module):
    def __init__(self, input_dim=17, num_filters=32, kernel_size=3, hidden_dim=64, num_layers=2, output_dim=2):
        super(TrafficCNNLSTMModel, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(num_filters, hidden_dim, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim) -> transpose to (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        # transpose back to (batch_size, seq_len, num_filters)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
