import configargparse
from data import load_test, preprocess
import ids
from sklearn import metrics
from tabulate import tabulate
from train_ids import get_model, parse_arguments

def main():
    options = parse_arguments()
    scores = test(options)
    print_scores(scores)

def test(options):
    attributes, labels = preprocess(load_test(), normalize=options.normalize)
    n_attributes = attributes.shape[1]
    model = get_model(options, n_attributes)
    model.load(options.save_model)
    predictions = model.predict(attributes)
    return get_scores(labels, predictions)

def get_scores(labels, predictions):
    accuracy = metrics.accuracy_score(labels, predictions)
    f1 = metrics.f1_score(labels, predictions, average='micro')
    precision = metrics.precision_score(labels, predictions, average='micro')
    recall = metrics.recall_score(labels, predictions, average='micro')
    return accuracy, f1, precision, recall

def print_scores(scores):
    scores = list(map(lambda score: f'{score:0.4f}', scores))
    headers = ['accuracy', 'f1', 'precision', 'recall']
    print(tabulate([scores], headers=headers))

if __name__ == '__main__':
    main()
