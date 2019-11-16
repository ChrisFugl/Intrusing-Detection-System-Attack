import configargparse
from data import load_test, preprocess
import ids
from train_ids import get_model, parse_arguments
from scores import get_binary_class_scores, print_scores
import sys

def main():
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    scores = test(options)
    print_scores(scores)

def test(options):
    attributes, labels = preprocess(load_test(), normalize=options.normalize)
    n_attributes = attributes.shape[1]
    model = get_model(options, n_attributes)
    model.load(options.save_model)
    predictions = model.predict(attributes)
    return get_binary_class_scores(labels, predictions)

if __name__ == '__main__':
    main()
