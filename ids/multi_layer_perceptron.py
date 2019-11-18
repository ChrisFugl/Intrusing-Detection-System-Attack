from collections import OrderedDict
from ids.abstract_model import AbstractModel
from scores import get_binary_class_scores
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils import with_cpu, with_gpu

class MultiLayerPerceptron(AbstractModel):
    """
    Multi-layer perceptron.
    """

    def __init__(self, input_size,
                log_dir, log_every=20, evaluate_every=100,
                epochs=10, batch_size=128, learning_rate=0.001,
                weight_decay=0, dropout_rate=0.5, hidden_size=128):
        output_size = 1
        self.model = with_gpu(nn.Sequential(OrderedDict([
            ('hidden1', nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)),
            ('hidden1_batchnorm', nn.BatchNorm1d(hidden_size)),
            ('hidden1_dropout', nn.Dropout(p=dropout_rate)),
            ('hidden1_activation', nn.ReLU()),
            ('hidden2', nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)),
            ('hidden2_batchnorm', nn.BatchNorm1d(hidden_size)),
            ('hidden2_dropout', nn.Dropout(p=dropout_rate)),
            ('hidden2_activation', nn.ReLU()),
            ('output', nn.Linear(in_features=hidden_size, out_features=output_size, bias=True))
        ])))
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.log_every = log_every
        self.evaluate_every = evaluate_every
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains multilayer perceptron.

        :param X: N x input_size ndarray
        :param y: N x 1 ndarray
        """
        writer_train = SummaryWriter(self.log_dir + 'train/')
        writer_val = SummaryWriter(self.log_dir + 'val/')
        self.model.train()
        n_observations = len(X_train)
        total_batches = n_observations // self.batch_size
        X_train_tensor, y_train_tensor = self.numpy2tensor(X_train, y_train)
        X_val_tensor, y_val_tensor = self.numpy2tensor(X_val, y_val)
        writer_train.add_graph(self.model, X_train_tensor)
        batches = 0
        iterations = 0
        for epoch in range(self.epochs):
            for batch_number in range(total_batches):
                batch_X, batch_y = self.get_batch(X_train_tensor, y_train_tensor, batch_number)
                logits_train = self.input2logits(batch_X)
                self.optimize(logits_train, batch_y)

                if batches % self.log_every == 0:
                    self.log(writer_train, iterations, logits_train, batch_y)

                if batches % self.evaluate_every == 0:
                    self.model.eval()
                    logits_val = self.input2logits(X_val_tensor)
                    self.model.train()
                    self.log(writer_val, iterations, logits_val, y_val_tensor)

                batches += 1
                iterations += self.batch_size
        writer_train.close()
        writer_val.close()

    def numpy2tensor(self, X, y):
        X_tensor = with_gpu(torch.from_numpy(X).float())
        y_tensor = with_gpu(torch.from_numpy(y).float())
        return X_tensor, y_tensor

    def get_batch(self, X, y, batch_number):
        batch_start = batch_number * self.batch_size
        batch_finish = (batch_number + 1) * self.batch_size
        batch_X = X[batch_start:batch_finish]
        batch_y = y[batch_start:batch_finish]
        return batch_X, batch_y

    def input2logits(self, X):
        return self.model(X).squeeze()

    def optimize(self, logits, labels):
        self.optimizer.zero_grad()
        loss = self.criterion(logits, labels)
        loss.backward()
        self.optimizer.step()

    def log(self, writer, iterations, logits_tensor, labels_tensor):
        loss = self.criterion(logits_tensor, labels_tensor)
        predictions = self.logits2prediction(logits_tensor)
        labels = with_cpu(labels_tensor).numpy()
        accuracy, f1, precision, recall, detection_rate = get_binary_class_scores(labels, predictions)
        writer.add_scalar('loss', loss.item(), iterations)
        writer.add_scalar('scores/accuracy', accuracy, iterations)
        writer.add_scalar('scores/f1', f1, iterations)
        writer.add_scalar('scores/precision', precision, iterations)
        writer.add_scalar('scores/recall', recall, iterations)
        writer.add_scalar('scores/detection_rate', detection_rate, iterations)
        writer.flush()

    def predict(self, X):
        self.model.eval()
        X_tensor = with_gpu(torch.from_numpy(X).float())
        logits_tensor = self.input2logits(X_tensor)
        return self.logits2prediction(logits_tensor)

    def logits2prediction(self, logits_tensor):
        predictions = torch.empty_like(logits_tensor)
        predictions[logits_tensor < 0] = 0
        predictions[logits_tensor >= 0] = 1
        return with_cpu(predictions).numpy()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
