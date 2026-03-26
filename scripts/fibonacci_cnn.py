import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class FibonacciCNN(nn.Module):
    def __init__(self):
        super(FibonacciCNN, self).__init__()
        # input shape: (batch, channels=1, length)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=2)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        # x: (batch, 1, length)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        # Global Average Pooling
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        return x

def create_cnn_sequences(data, window_size=4):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def predict_next_numbers(initial_sequence, n_preds):
    window_size = 4
    data = np.array(initial_sequence, dtype=np.float32)

    # Simple training on the fly for this exercise
    X_train, y_train = create_cnn_sequences(data, window_size)

    # If initial sequence is too short, we might need a smaller window or more data
    # But [1, 1, 2, 3, 5, 8, 13, 21] is length 8, so window 4 gives 4 samples.

    X_tensor = torch.tensor(X_train).unsqueeze(1) # (samples, 1, window_size)
    y_tensor = torch.tensor(y_train).unsqueeze(1) # (samples, 1)

    model = FibonacciCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

    # Multi-step prediction
    model.eval()
    current_seq = list(data)
    with torch.no_grad():
        for _ in range(n_preds):
            inp = torch.tensor(current_seq[-window_size:], dtype=torch.float32).view(1, 1, window_size)
            pred = model(inp)
            current_seq.append(round(pred.item()))

    return current_seq
