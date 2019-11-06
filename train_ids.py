import configargparse
from data import load_train, preprocess
import ids
import yaml

def main():
    options = parse_arguments()
    attributes, attack_class = load_data(load_train(), options)
    model = get_model(options)
    model.train(attributes, attack_class)

    # save model
    if options.save_model is not None:
        model.save(options.save_model)

def parse_arguments():
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('--config', required=False, is_config_file=True, help='config file path')
    parser.add('--save_config', required=False, default=None, type=str, help='path of config file where arguments can be saved')
    parser.add('--save_model', required=False, default=None, type=str, help='path of file to save trained model')
    parser.add('--algorithm', required=True, choices=['dt', 'knn', 'lr', 'mlp', 'nb', 'rf', 'svm'], help='algorithm to train')
    parser.add('--normalize', required=False, action='store_true', default=False, help='normalize data (default false)')
    parser.add('--iterations', required=False, type=int, default=1000, help='number of training iterations (default 1000)')
    parse_ids_arguments(parser)
    options = parser.parse_args()

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
    knn_group.add('--n_neighbors', required=False, default=5, type=int, help='number of neighbours to compare')

def load_data(data, options):
    attributes_dataframe, _, attack_class_dataframe = preprocess(data, normalize=options.normalize)
    attributes = attributes_dataframe.to_numpy()
    attack_class = attack_class_dataframe.to_numpy()
    return attributes, attack_class

def get_model(options):
    algorithm = options.algorithm
    if algorithm == 'dt':
        return ids.DecisionTree()
    elif algorithm == 'knn':
        return ids.KNearestNeighbours(n_neighbors=options.n_neighbors)
    elif algorithm == 'lr':
        return ids.LogisticRegression(max_iter=options.iterations)
    elif algorithm == 'mlp':
        raise NotImplementedError(algorithm)
    elif algorithm == 'nb':
        return ids.NaiveBayes()
    elif algorithm == 'rf':
        raise NotImplementedError(algorithm)
    elif ids == 'svm':
        raise NotImplementedError(algorithm)
    else:
        raise NotImplementedError(f'"{algorithm}" is not a valid choice of algorithm.')

if __name__ == '__main__':
    main()
