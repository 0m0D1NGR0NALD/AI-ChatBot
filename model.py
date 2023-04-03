import torch.nn as nn

def __init__(self, input_size, embedding_size, hidden_size, output_size):
    super(NeuralNet, self).__init__()
    self.embedding = nn.Embedding(input_size, embedding_size)
    self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)
