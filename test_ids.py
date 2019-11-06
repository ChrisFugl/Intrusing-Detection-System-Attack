import configargparse
from data import load_test
import ids
from sklearn import metrics
from tabulate import tabulate
from train_ids import get_model, load_data, parse_arguments

def main():
    options = parse_arguments()
    scores = test(options)
    print_scores(scores)

def test(options):
    attributes, attack_class = load_data(load_test(), options)
    model = get_model(options)
    model.load(options.save_model)
    predictions = model.predict(attributes)
    return get_scores(attack_class, predictions)

def get_scores(attack_class, predictions):
    accuracy = metrics.accuracy_score(attack_class, predictions)
    f1 = metrics.f1_score(attack_class, predictions, average='micro')
    precision = metrics.precision_score(attack_class, predictions, average='micro')
    recall = metrics.recall_score(attack_class, predictions, average='micro')
    return accuracy, f1, precision, recall

def print_scores(scores):
    scores = list(map(lambda score: f'{score:0.4f}', scores))
    headers = ['accuracy', 'f1', 'precision', 'recall']
    print(tabulate([scores], headers=headers))

if __name__ == '__main__':
    main()
