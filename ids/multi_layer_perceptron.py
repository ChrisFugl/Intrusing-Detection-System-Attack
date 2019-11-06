from collections import OrderedDict
from ids.abstract_model import AbstractModel
import torch
import torch.nn as nn

class MultiLayerPerceptron(AbstractModel):
    """
    Multi-layer perceptron.
    """

    def __init__(self, input_size, epochs=10, batch_size=128, learning_rate=0.001, weight_decay=0, dropout_rate=0.5, hidden_size=128):
        output_size = 1
        self.model = nn.Sequential(OrderedDict([
            ('hidden', nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)),
            ('hidden_activation', nn.ReLU()),
            ('output_batchnorm', nn.BatchNorm1d(hidden_size)),
            ('output_dropout', nn.Dropout(p=dropout_rate)),
            ('output', nn.Linear(in_features=hidden_size, out_features=output_size, bias=True))
        ]))
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def train(self, X, y):
        """
        Trains multilayer perceptron.

        :param X: N x input_size ndarray
        :param y: N x 1 ndarray
        """
        self.model.train()
        n_observations = len(X)
        total_batches = n_observations // self.batch_size
        for epoch in range(self.epochs):
            for batch_number in range(total_batches):
                batch_start = batch_number * self.batch_size
                batch_finish = (batch_number + 1) * self.batch_size
                batch_X = torch.from_numpy(X[batch_start:batch_finish]).float()
                batch_y = torch.from_numpy(y[batch_start:batch_finish]).float()
                logits = self.model(batch_X).squeeze()
                self.optimizer.zero_grad()
                loss = self.criterion(logits, batch_y)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        self.model.eval()
        batch = torch.from_numpy(X).float()
        outputs = self.model(batch)
        predictions = torch.empty_like(outputs)
        predictions[outputs < 0] = 0
        predictions[outputs >= 0] = 1
        return predictions.numpy()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
