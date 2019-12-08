from functools import partial
import numpy as np
from operator import itemgetter
import pandas as pd
from sklearn import preprocessing

# features
_INTRINSIC = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent'
]
_CONTENT = [
    'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login'
]
_TIME_BASED = [
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate'
]
_HOST_BASED = [
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]
_LABELS = ['class', 'difficulty_level']

_HEADERS = _INTRINSIC + _CONTENT + _TIME_BASED + _HOST_BASED + _LABELS

# datatypes used to convert some features
_DTYPES = {
    'protocol_type': pd.CategoricalDtype(['tcp', 'udp', 'icmp']),
    'service': pd.CategoricalDtype([
        'aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime',
        'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs',
        'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames',
        'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC',
        'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp',
        'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat',
        'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3',
        'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell',
        'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet',
        'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path',
        'vmnet', 'whois', 'X11', 'Z39_50'
    ]),
    'flag': pd.CategoricalDtype([
        'OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR',
        'S0', 'S1', 'S2', 'S3', 'SF', 'SH'
    ]),
    'class': pd.CategoricalDtype([
        'normal', 'neptune', 'warezclient', 'ipsweep', 'portsweep', 'teardrop',
        'nmap', 'satan', 'smurf', 'pod', 'back', 'guess_passwd', 'ftp_write',
        'multihop', 'rootkit', 'buffer_overflow', 'imap', 'warezmaster', 'phf',
        'land', 'loadmodule', 'spy', 'perl'
    ]),
    'land': np.bool,
    'logged_in': np.bool,
    'is_host_login': np.bool,
    'is_guest_login': np.bool,
    'difficulty_level': pd.CategoricalDtype(list(range(22)), ordered=True)
}

def load_train():
    """
    Loads trainingset as a Pandas dataframe.
    """
    return _add_attack_class(
        pd.read_csv(
            'data/KDDTrain.csv',
            header=None,
            names=_HEADERS,
            dtype=_DTYPES
        )
    )

def load_test():
    """
    Loads testset as a Pandas dataframe.
    """
    return _add_attack_class(
        pd.read_csv(
            'data/KDDTest.csv',
            header=None,
            names=_HEADERS,
            dtype=_DTYPES
        )
    )

def load_val():
    """
    Loads validationset as a Pandas dataframe.
    """
    return _add_attack_class(
        pd.read_csv(
            'data/KDDVal.csv',
            header=None,
            names=_HEADERS,
            dtype=_DTYPES
        )
    )

def _add_attack_class(data):
    """
    Adds attack_class column to dataframe.
    """
    dtype = pd.CategoricalDtype(['Normal', 'DoS', 'R2L', 'U2R', 'Probe'])
    dos = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop']
    r2l = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster']
    u2r = ['buffer_overflow', 'loadmodule', 'perl', 'rootkit']
    probe = ['ipsweep', 'nmap', 'portsweep', 'satan']
    column = pd.Series(['Normal'] * len(data), dtype=dtype)
    column[data['class'].isin(dos)] = 'DoS'
    column[data['class'].isin(r2l)] = 'R2L'
    column[data['class'].isin(u2r)] = 'U2R'
    column[data['class'].isin(probe)] = 'Probe'
    data['attack_class'] = column
    return data

def preprocess(dataframe, normalize=False, remove_classes=['R2L', 'U2R']):
    """
    Preprocesses data.
    Performs these operations:
    * Removes rows with attack classes specified in argument: (default R2L, U2R).
    * Normalize numeric attributes unless specified otherwise.
    * Splits data into: attributes, attack_class.
    * One-hot encodes columns: protocol_type, service, flag, class, attack_class.
    * Makes attack class binary (in accordance with the selected attack class).
    :param dataframe: data
    :param normalize: flag if the numerical attributes should be normalized (default false)
    :param one_hot_encode_labels: flag if the labels should be one hot encoded (default false)
    :param remove_classes: rows with one of these attack classes will be removed (default ['R2L', 'U2R'])
    :return: attributes, binary attack class
    :rtype: (ndarray, ndarray)
    """
    preprocessed = dataframe

    label_columns = ['class', 'attack_class']
    categorical_columns = ['protocol_type', 'service', 'flag']
    boolean_columns = ['land', 'logged_in', 'is_host_login', 'is_guest_login']
    # difficulty_level is a categorical column, but we do not want to one-hot encode it, and neither should it be considered as numeric
    numeric_columns = list(
        set(preprocessed.columns)
      - set(label_columns)
      - set(categorical_columns)
      - set(boolean_columns)
      - set(['difficulty_level'])
    )
    categorical_columns = list(set(categorical_columns) & set(preprocessed.columns))
    label_columns = list(set(label_columns) & set(preprocessed.columns))
    boolean_columns = list(set(boolean_columns) & set(preprocessed.columns))

    # remove attack classes
    if len(remove_classes) != 0:
        removed_classes_index = preprocessed[preprocessed['attack_class'].isin(remove_classes)].index
        preprocessed = preprocessed.drop(index=removed_classes_index).reset_index(drop=True)

    #attributes_dataframe = preprocessed.drop(columns=['class', 'attack_class', 'difficulty_level'])
    #attack_class_dataframe = preprocessed['attack_class']
    #print(attributes_dataframe.iloc[0])

    # normalize
    
    if normalize:
        # laod trainingset since the numeric columns should be standardized in accordance with the trainingset,
        # and we do not know if the dataframe represents the trainingset
        training = pd.read_csv('data/KDDTrain.csv', header=None, names=_HEADERS, usecols=numeric_columns)
        mean = training.mean(axis=0)
        std = training.std(axis=0)

        # set to zero where std is zero
        zero_std_columns = std == 0
        zero_std_columns = zip(zero_std_columns.index, zero_std_columns)
        zero_std_columns = filter(itemgetter(1), zero_std_columns)
        zero_std_columns = list(map(itemgetter(0), zero_std_columns))
        non_zero_std_columns = list(set(numeric_columns) - set(zero_std_columns))

        preprocessed[zero_std_columns] = 0
        preprocessed[non_zero_std_columns] = (preprocessed[non_zero_std_columns] - mean[non_zero_std_columns]) / std[non_zero_std_columns]
        #min_max_scaler = preprocessing.MinMaxScaler()
        #attributes_dataframe = min_max_scaler.fit_transform(attributes_dataframe)

    # split into (attributes, attack_class) and remove class from attributes
    attributes_dataframe = preprocessed.drop(columns=['class', 'attack_class', 'difficulty_level'])
    attack_class_dataframe = preprocessed['attack_class']
    #print(attributes_dataframe.iloc[0])

    # one-hot encoding
    attributes_dataframe = pd.get_dummies(attributes_dataframe, columns=categorical_columns)
    #print(attributes_dataframe.iloc[0])

    # make attack class binary (0 = normal, 1 = malicious)
    binary_attack_class = np.zeros_like(attack_class_dataframe, dtype=np.bool)
    binary_attack_class[attack_class_dataframe != 'Normal'] = 1

    attributes = attributes_dataframe.to_numpy().astype(np.float)
    #attributes = attributes_dataframe.astype(np.float)
    return attributes, binary_attack_class

def split_features(dataframe, selected_attack_class):
    normal = dataframe[dataframe['attack_class'].isin(['Normal'])]
    normal_ff = normal
    normal_nff = remove_intrinsic(normal)

    features = dataframe[dataframe['attack_class'].isin([selected_attack_class])]
    #test = len(dataframe[dataframe['attack_class'].isin([selected_attack_class])])
    #print(features.iloc[0])
    #print(features.shape)
    functionnal_features = features
    non_functionnal_features = remove_intrinsic(features)

    # remove features based on attack class
    if selected_attack_class == 'DoS':
        functionnal_features = remove_content(functionnal_features)
        functionnal_features = remove_host_based(functionnal_features)
        non_functionnal_features = remove_time_based(non_functionnal_features)
        test = len(non_functionnal_features)
        #print(non_functionnal_features.shape)

        normal_ff = remove_content(normal_ff)
        normal_ff = remove_host_based(normal_ff)
        normal_nff = remove_time_based(normal_nff)
    elif selected_attack_class == 'Probe':
        functionnal_features = remove_content(functionnal_features)
        non_functionnal_features = remove_host_based(non_functionnal_features)
        non_functionnal_features = remove_time_based(non_functionnal_features)

        normal_ff = remove_content(normal_ff)
        normal_nff = remove_host_based(normal_nff)
        normal_nff = remove_time_based(normal_nff)

    return functionnal_features, non_functionnal_features, normal_ff, normal_nff

def get_content_columns():
    """
    Returns the content column names.
    """
    return _CONTENT

def get_host_based_columns():
    """
    Returns the host based column names.
    """
    return _HOST_BASED

def get_time_based_columns():
    """
    Returns the time based column names.
    """
    return _TIME_BASED

def remove_content(dataframe):
    """
    Removes all content features from the dataframe.
    """
    return dataframe.drop(columns=_CONTENT).reset_index(drop=True)

def remove_time_based(dataframe):
    """
    Removes all time based features from the dataframe.
    """
    return dataframe.drop(columns=_TIME_BASED).reset_index(drop=True)

def remove_host_based(dataframe):
    """
    Removes all host based features from the dataframe.
    """
    return dataframe.drop(columns=_HOST_BASED).reset_index(drop=True)

def remove_intrinsic(dataframe):
    """
    Removes all host based features from the dataframe.
    """
    return dataframe.drop(columns=_INTRINSIC).reset_index(drop=True)
