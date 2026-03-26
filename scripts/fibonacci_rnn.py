import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def predict_next_number(sequence):
    seq_array = np.array(sequence, dtype=np.float32)

    # ── ONLY THIS CLASS IS DIFFERENT ──────────────────────────────
    class RNNPredictor(nn.Module):
        def __init__(self, proj_dim: int = 4, hidden_dim: int = 10):
            super().__init__()
            self.pre = nn.Linear(1, proj_dim)                     # ⟨1⟩ NEW
            self.rnn = nn.RNN(                                    # ⟨2⟩ change input_size
                input_size  = proj_dim,
                hidden_size = hidden_dim,
                num_layers  = 1,
                batch_first = True
            )
            self.fc  = nn.Linear(hidden_dim, 1)

        def forward(self, x, hidden):
            x = self.pre(x)                                       # ⟨3⟩ NEW
            out, hidden = self.rnn(x, hidden)
            out = self.fc(out[:, -1, :])
            return out, hidden
    # ─────────────────────────────────────────────────────────────

    def create_sequences(data):
        X = np.array(data).reshape(1, -1, 1)                       # full sequence
        Y = np.array(data[-1] + data[-2]).reshape(1, 1)            # next Fib
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

    inputs, targets = create_sequences(seq_array)

    model     = RNNPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop (30 k epochs like before)
    epochs = 30_000
    for _ in range(epochs):
        optimizer.zero_grad()
        hidden = torch.zeros(1, 1, 10)
        output, _ = model(inputs, hidden)
        loss = criterion(output.squeeze(), targets.squeeze())
        loss.backward()
        optimizer.step()

    # One‑step forecast (unchanged)
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([[sequence[-1]]],
                                    dtype=torch.float32).view(1, 1, 1)
        hidden = torch.zeros(1, 1, 10)
        prediction, _ = model(input_tensor, hidden)

    return round(prediction.item(), 0)

# ── driver (unchanged) ──────────────────────────────────────────
if __name__ == "__main__":
    training_sequence = np.array([1, 1, 2, 3, 5, 8, 13, 21], dtype=np.float32)
    predicted_fibonacci = np.array([])

    for i in range(1, 6):
        next_fib = predict_next_number(training_sequence)
        print(f"Predicted next Fibonacci number ({8 + i}):", next_fib)
        training_sequence = np.append(training_sequence, next_fib)
        predicted_fibonacci = np.append(predicted_fibonacci, next_fib)
