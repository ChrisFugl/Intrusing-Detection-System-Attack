import configargparse
from data import load_train, preprocess
import ids
import yaml

def main():
    options = parse_arguments()
    attributes, attack_class = load_data(options)
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
    parser.add('--attack_class', required=False, default=None, choices=['dos', 'probe', 'r2l'], help='selects features based on the attack class (defaults to select all features)')
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

def load_data(options):
    attributes_dataframe, _, attack_class_dataframe = preprocess(load_train(), selected_attack_class=options.attack_class)
    attributes = attributes_dataframe.to_numpy()
    attack_class = attack_class_dataframe.to_numpy()
    return attributes, attack_class

def get_model(options):
    algorithm = options.algorithm
    if algorithm == 'dt':
        return ids.DecisionTree()
    elif algorithm == 'knn':
        raise Exception(f'Not implemented yet ({algorithm}).')
    elif algorithm == 'lr':
        raise Exception(f'Not implemented yet ({algorithm}).')
    elif algorithm == 'mlp':
        raise Exception(f'Not implemented yet ({algorithm}).')
    elif algorithm == 'nb':
        raise Exception(f'Not implemented yet ({algorithm}).')
    elif algorithm == 'rf':
        raise Exception(f'Not implemented yet ({algorithm}).')
    elif ids == 'svm':
        raise Exception(f'Not implemented yet ({algorithm}).')
    else:
        raise Exception(f'"{algorithm}" is not a valid choice of algorithm.')

if __name__ == '__main__':
    main()
