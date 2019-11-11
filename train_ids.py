import configargparse
from data import load_train, preprocess
import ids
import numpy as np
import sys
import yaml

def main():
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    attributes, labels = preprocess(load_train(), normalize=options.normalize)
    n_attributes = attributes.shape[1]
    model = get_model(options, n_attributes)
    model.train(attributes, labels)

    # save model
    if options.save_model is not None:
        model.save(options.save_model)

def parse_arguments(arguments):
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('--config', required=False, is_config_file=True, help='config file path')
    parser.add('--save_config', required=False, default=None, type=str, help='path of config file where arguments can be saved')
    parser.add('--save_model', required=False, default=None, type=str, help='path of file to save trained model')
    parser.add('--algorithm', required=True, choices=['baseline', 'dt', 'knn', 'lr', 'mlp', 'nb', 'rf', 'svm'], help='algorithm to train')
    parser.add('--normalize', required=False, action='store_true', default=False, help='normalize data (default false)')
    parser.add('--iterations', required=False, type=int, default=1000, help='number of training iterations (default 1000)')
    parse_ids_arguments(parser)
    options = parser.parse_args(arguments)

    # remove keys that should not be saved to config file
    save_config = options.save_config
    del options.config
    del options.save_config

    # save config file
    if save_config is not None:
        with open(save_config, 'w') as config_file:
            yaml.dump(vars(options), config_file)

    return options

def parse_ids_arguments(parser):
    knn_group = parser.add_argument_group('knn')
    knn_group.add('--n_neighbors', required=False, default=5, type=int, help='number of neighbours to compare (default 5)')

    tree_group = parser.add_argument_group('trees (decision tree and random forest)')
    tree_group.add('--n_trees', required=False, default=10, type=int, help='number of trees in a random forest (default 5)')
    tree_group.add('--max_depth', required=False, default=None, type=null_or_int, help='maximum tree depth (default infinite)')
    tree_group.add('--split_criterion', required=False, default='gini', choices=['gini', 'entropy'], help='criterion for how to split a node (default gini)')
    tree_group.add('--min_samples_leaf', required=False, default=1, type=int, help='minimum number of samples required for a node to be a leaf (default 1)')
    tree_group.add('--min_samples_split', required=False, default=2, type=int, help='minimum number of samples required to split a node (default 2)')

    mlp_group = parser.add_argument_group('mlp')
    mlp_group.add('--epochs', required=False, default=10, type=int, help='epochs of training (default 10)')
    mlp_group.add('--batch_size', required=False, default=128, type=int, help='batch size (default 128)')
    mlp_group.add('--learning_rate', required=False, default=0.001, type=float, help='learning rate (default 0.001)')
    mlp_group.add('--weight_decay', required=False, default=0, type=float, help='weight decay/L2 regularization strength (default 0)')
    mlp_group.add('--dropout_rate', required=False, default=0.5, type=float, help='dropout rate (default 0.5)')
    mlp_group.add('--hidden_size', required=False, default=128, type=int, help='hidden layer size')

def null_or_int(value):
    if value is None or value == 'null' or value == 'None':
        return None
    else:
        return int(value)

def get_model(options, n_features):
    algorithm = options.algorithm
    if algorithm == 'baseline':
        return ids.Baseline()
    elif algorithm == 'dt':
        return ids.DecisionTree(
            max_depth=options.max_depth,
            split_criterion=options.split_criterion,
            min_samples_leaf=options.min_samples_leaf,
            min_samples_split=options.min_samples_split
        )
    elif algorithm == 'knn':
        return ids.KNearestNeighbours(n_neighbors=options.n_neighbors)
    elif algorithm == 'lr':
        return ids.LogisticRegression(max_iter=options.iterations)
    elif algorithm == 'mlp':
        return ids.MultiLayerPerceptron(
            input_size=n_features,
            epochs=options.epochs,
            batch_size=options.batch_size,
            learning_rate=options.learning_rate,
            weight_decay=options.weight_decay,
            dropout_rate=options.dropout_rate,
            hidden_size=options.hidden_size
        )
    elif algorithm == 'nb':
        return ids.NaiveBayes()
    elif algorithm == 'rf':
        return ids.RandomForest(
            n_trees=options.n_trees,
            max_depth=options.max_depth,
            split_criterion=options.split_criterion,
            min_samples_leaf=options.min_samples_leaf,
            min_samples_split=options.min_samples_split
        )
    elif algorithm == 'svm':
        return ids.SupportVectorMachine(max_iter=options.iterations)
    else:
        raise Exception(f'"{algorithm}" is not a valid choice of algorithm.')

if __name__ == '__main__':
    main()
