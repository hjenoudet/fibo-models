import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def predict_next_number(sequence):
    seq_array = np.array(sequence, dtype=np.float32)

    class RNNPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.RNN(input_size=1, hidden_size=10, num_layers=1, batch_first=True) # One Hidden Layer, 10 neurons as per the requirements on the homework.
            self.fc = nn.Linear(10, 1)

        def forward(self, x, hidden):
            out, hidden = self.rnn(x, hidden)  # RNN processing
            out = self.fc(out[:, -1, :])
            return out, hidden

    def create_sequences(data):
        X = np.array(data).reshape(1, -1, 1) # Convert full sequence to input
        Y = np.array(data[-1] + data[-2]).reshape(1, 1)  # Next Fibonacci number as target

        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

    # Crafting training data using the full input sequence
    inputs, targets = create_sequences(seq_array)

    # Initializing our RNN (low learning rate and Adam, as per the homework requirements)
    model = RNNPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    epochs = 30000
    for epoch in range(epochs):
        optimizer.zero_grad()
        hidden = torch.zeros(1, 1, 10)  # Initialize hidden state
        output, _ = model(inputs, hidden)  # Forward pass
        loss = criterion(output.squeeze(), targets.squeeze()) # Loss
        loss.backward() # Backpropagation
        optimizer.step() # Optimization

    # Predict the Next Number Using ONLY the last number of the sequence, as per the restrictions.
    model.eval()
    with torch.no_grad():
      input_tensor = torch.tensor([[sequence[-1]]], dtype=torch.float32).view(1, 1, 1) # Use only one word to predict the next few numbers
      hidden = torch.zeros(1, 1, 10)  # Initialization of hidden state for prediction
      prediction, _ = model(input_tensor, hidden)

    return round(prediction.item(), 0)  # Return single predicted number

if __name__ == "__main__":
    training_sequence = np.array([1, 1, 2, 3, 5, 8, 13, 21], dtype=np.float32)  # We use the first 8 numbers of the Fibonacci sequence for training our RNN
    predicted_fibonacci = np.array([])
    for i in range(1,6):
      next_fib = predict_next_number(training_sequence)
      print(f"Predicted next Fibonacci number ({8+i}):", next_fib) # We print the RNN's prediction, which outputs one number, as per the restrictions
      training_sequence = np.append(training_sequence, next_fib) # We add the prediction at the end of our training sequence, and use that new training sequence for the next prediction
      predicted_fibonacci = np.append(predicted_fibonacci, next_fib) # we keep the predicted values separated for further model accuracy analysis.
