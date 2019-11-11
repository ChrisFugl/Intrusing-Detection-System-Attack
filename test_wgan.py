import configargparse
from data import load_test, preprocess
from model import WGAN
from train_wgan import parse_arguments
from scores import get_binary_class_scores, print_scores

def main():
    options = parse_arguments()
    scores = test(options)
    print_scores(scores)

def test(options):
    M_attributes, labels = preprocess(load_test(), type="Malicious", normalize=options.normalize)
    n_attributes = M_attributes.shape[1]
    model = WGAN(options, n_attributes)
    model.load(options.save_model)
    predictions, labels = model.predict(M_attributes, labels)
    return get_binary_class_scores(labels, predictions)

def print_scores(scores):
    scores = list(map(lambda score: f'{score:0.4f}', scores))
    headers = ['accuracy', 'f1', 'precision', 'recall']
    print(tabulate([scores], headers=headers))

if __name__ == '__main__':
    main()
