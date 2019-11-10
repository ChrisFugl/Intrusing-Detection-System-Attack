import configargparse
from data import load_train, preprocess
from model import WGAN
import yaml

def main():
    options = parse_arguments()
    N_attributes, _ = preprocess(load_train(), type="Normal", normalize=options.normalize)
    M_attributes, _ = preprocess(load_train(), type="Malicious", normalize=options.normalize)
    n_attributes = N_attributes.shape[1]
    model = WGAN(options, n_attributes)
    model.train(N_attributes, M_attributes)

    # save model
    if options.save_model is not None:
        model.save(options.save_model)

def parse_arguments():
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('--config', required=False, is_config_file=True, help='config file path')
    parser.add('--save_config', required=False, default=None, type=str, help='path of config file where arguments can be saved')
    parser.add('--save_model', required=False, default=None, type=str, help='path of file to save trained model')
    parser.add('--normalize', required=False, action='store_true', default=False, help='normalize data (default false)')
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
    wgan_group.add('--epochs', required=False, default=100, type=int, help='epochs of training (default 100)')
    wgan_group.add('--batch_size', required=False, default=64, type=int, help='batch size (default 64)')
    wgan_group.add('--learning_rate', required=False, default=0.0001, type=float, help='learning rate (default 0.0001)')
    wgan_group.add('--weight_clipping', required=False, default=0.01, type=float, help='weight clipping threshold (default 0.01)')
    wgan_group.add('--noise_dim', required=False, default=9, type=int, help='dimension of the noise vector (default 9)')
    wgan_group.add('--critic_iter', required=False, default=5, type=int, help='Number of critic iteration per epoch (default 5)')

if __name__ == '__main__':
    main()