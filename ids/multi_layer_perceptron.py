from collections import OrderedDict
from ids.abstract_model import AbstractModel
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class MultiLayerPerceptron(AbstractModel):
    """
    Multi-layer perceptron.
    """

    def __init__(self, input_size,
                log_dir, log_every=1000, evaluate_every=10000,
                epochs=10, batch_size=128, learning_rate=0.001, weight_decay=0, dropout_rate=0.5, hidden_size=128):
        output_size = 1
        self.model = nn.Sequential(OrderedDict([
            ('hidden1', nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)),
            ('hidden1_batchnorm', nn.BatchNorm1d(hidden_size)),
            ('hidden1_dropout', nn.Dropout(p=dropout_rate)),
            ('hidden1_activation', nn.ReLU()),
            ('hidden2', nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)),
            ('hidden2_batchnorm', nn.BatchNorm1d(hidden_size)),
            ('hidden2_dropout', nn.Dropout(p=dropout_rate)),
            ('hidden2_activation', nn.ReLU()),
            ('output', nn.Linear(in_features=hidden_size, out_features=output_size, bias=True))
        ]))
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.log_dir = log_dir
        self.log_every = log_every
        self.evaluate_every = evaluate_every

    def train(self, X, y):
        """
        Trains multilayer perceptron.

        :param X: N x input_size ndarray
        :param y: N x 1 ndarray
        """
        writer_train = SummaryWriter(self.log_dir + 'train/')
        self.model.train()
        n_observations = len(X)
        total_batches = n_observations // self.batch_size
        X_torch = torch.from_numpy(X).float()
        y_torch = torch.from_numpy(y).float()
        writer_train.add_graph(self.model, X_torch)
        iterations = 0
        for epoch in range(self.epochs):
            for batch_number in range(total_batches):
                batch_start = batch_number * self.batch_size
                batch_finish = (batch_number + 1) * self.batch_size
                batch_X = X_torch[batch_start:batch_finish]
                batch_y = y_torch[batch_start:batch_finish]
                logits = self.model(batch_X).squeeze()

                self.optimizer.zero_grad()
                loss = self.criterion(logits, batch_y)
                loss.backward()
                self.optimizer.step()

                if iterations % self.log_every < self.batch_size:
                    self.log(writer_train, iterations, loss, logits, batch_y)

                iterations += self.batch_size
        writer_train.close()

    def log(self, writer, iterations, loss, logits, labels):
        writer.add_scalar('loss', loss.item(), iterations)

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
