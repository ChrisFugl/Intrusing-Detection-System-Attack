import configargparse
from data import load_train, load_val, preprocess, split_features
from model import WGAN
import os
import yaml

def main():
    options = parse_arguments()
    functional_features, non_functional_features, normal_ff, normal_nff = split_features(load_train(), selected_attack_class=options.attack)
    nff_attributes, labels_mal = preprocess(non_functional_features, normalize=options.normalize)
    normal_attributes, labels_nor = preprocess(normal_nff, normalize=options.normalize)
    n_attributes = nff_attributes.shape[1]
    trainingset = (normal_attributes, nff_attributes, labels_nor, labels_mal)

    functional_features, non_functional_features, normal_ff, normal_nff = split_features(load_val(), selected_attack_class=options.attack)
    nff_attributes, labels_mal = preprocess(non_functional_features, normalize=options.normalize)
    normal_attributes, labels_nor = preprocess(normal_nff, normalize=options.normalize)
    n_attributes = nff_attributes.shape[1]
    validationset = (normal_attributes, nff_attributes, labels_nor, labels_mal)

    model = WGAN(options, n_attributes)
    model.train(trainingset, validationset)

    # save model
    if options.save_model is not None:
        save_model_directory = os.path.join(options.save_model, options.name)
        os.makedirs(save_model_directory, exist_ok=True)
        model.save(save_model_directory)

def parse_arguments():
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('--config', required=False, is_config_file=True, help='config file path')
    parser.add('--save_config', required=False, default=None, type=str, help='path of config file where arguments can be saved')
    parser.add('--save_model', required=False, default=None, type=str, help='path of file to save trained model')
    parser.add('--normalize', required=False, action='store_true', default=False, help='normalize data (default false)')
    parser.add('--attack', required=True, default='Probe', help='selected attack')
    parser.add('--name', required=True, type=str, help='Unique name of the experiment.')
    parser.add('--checkpoint', required=False, type=str, default=None, help='path to load checkpoint from')
    parser.add('--checkpoint_directory', required=False, type=str, default='checkpoints/', help='path to checkpoints directory (default: checkpoints/)')
    parser.add('--checkpoint_interval_s', required=False, type=int, default=1800, help='seconds between saving checkpoints (default: 1800)')
    parser.add('--evaluate', required=False, type=int, default=200, help='number of epochs between evaluating on validation set (default: 200)')
    parse_wgan_arguments(parser)
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

def parse_wgan_arguments(parser):
    wgan_group = parser.add_argument_group('wgan')
    wgan_group.add('--epochs', required=False, default=100, type=int, help='epochs of training (default 100), set to -1 to continue until manually stopped')
    wgan_group.add('--batch_size', required=False, default=64, type=int, help='batch size (default 64)')
    wgan_group.add('--learning_rate', required=False, default=0.0001, type=float, help='learning rate (default 0.0001)')
    wgan_group.add('--weight_clipping', required=False, default=0.01, type=float, help='weight clipping threshold (default 0.01)')
    wgan_group.add('--noise_dim', required=False, default=9, type=int, help='dimension of the noise vector (default 9)')
    wgan_group.add('--critic_iter', required=False, default=5, type=int, help='Number of critic iteration per epoch (default 5)')

if __name__ == '__main__':
    main()
