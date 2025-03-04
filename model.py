import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as transforms

class EmotionCNNLSTM(nn.Module):
    def __init__(self, num_classes=5, cnn_out_dim=64, lstm_hidden_size=128, num_lstm_layers=2):
        super(EmotionCNNLSTM, self).__init__()
        
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, cnn_out_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # LSTM for Temporal Dependencies
        self.lstm = nn.LSTM(input_size=cnn_out_dim, hidden_size=lstm_hidden_size, 
                            num_layers=num_lstm_layers, batch_first=True, bidirectional=True)
        
        # Fully Connected Layers for Classification
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Extract features using CNN
        x = self.cnn(x)  # Shape: [batch, cnn_out_dim, new_height, new_width]
        
        # Reshape for LSTM: (batch, seq_len, features)
        x = x.permute(0, 2, 3, 1).contiguous()  # (batch, height, width, channels)
        x = x.view(batch_size, -1, x.size(-1))  # Flatten spatial dimensions
        
        # Process with LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last output state
        
        # Classification
        x = self.fc(x)
        return x