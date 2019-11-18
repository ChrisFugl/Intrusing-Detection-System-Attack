import configargparse
from data import load_test, preprocess
import ids
import pandas as pd
from train_ids import get_model, parse_ids_arguments
from scores import get_binary_class_scores, print_scores
import sys

def main():
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    scores = test(options)
    print_scores(scores)

def test(options):
    attributes, labels = load_data(options)
    n_attributes = attributes.shape[1]
    model = get_model(options, n_attributes)
    model.load(options.save_model)
    predictions = model.predict(attributes)
    return get_binary_class_scores(labels, predictions)

def parse_arguments(arguments):
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('--algorithm', required=True, choices=['baseline', 'dt', 'knn', 'lr', 'mlp', 'nb', 'rf', 'svm'], help='algorithm to train')
    parser.add('--save_model', required=True, default=None, type=str, help='path of file to save trained model')
    parser.add('--config', required=False, is_config_file=True, help='config file path')
    parser.add('--attack', required=False, default=None, choices=['DoS', 'Probe', 'U2R_R2L'], help='select attack class to only evaluate on this attack class (default evaluate on all)')
    parser.add('--normalize', required=False, action='store_true', default=False, help='normalize data (default false)')
    parse_ids_arguments(parser)
    options = parser.parse_args(arguments)
    return options

def load_data(options):
    data = load_test()
    if options.attack is not None:
        attack_classes = get_attack_classes(options.attack)
        data = data[data.attack_class.isin(['Normal', *attack_classes])]
    return preprocess(data, normalize=options.normalize)

def get_attack_classes(attack):
    if attack == 'DoS':
        return ['DoS']
    elif attack == 'Probe':
        return ['Probe']
    else:
        return ['U2R', 'R2L']

if __name__ == '__main__':
    main()
