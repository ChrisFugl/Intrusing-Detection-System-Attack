import numpy as np
import pandas as pd

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
            header=0,
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
            header=0,
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

def preprocess(dataframe, normalize=True):
    """
    Preprocesses data.

    Performs these operations:
    * Removes rows with attack classes: U2R.
    * Normalize numeric attributes unless specified otherwise.
    * Splits data into: attributes, class, attack_class.
    * One-hot encodes columns: protocol_type, service, flag, class, attack_class.

    :param dataframe: Data
    :type dataframe: pd.DataFrame

    :param normalize: Flag if the numerical attributes should be normalized (default true)
    :type normalize: bool

    :return: attributes, class, attack_classs
    :rtype: (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    """
    label_columns = ['class', 'attack_class']
    categorical_columns = ['protocol_type', 'service', 'flag']
    boolean_columns = ['land', 'logged_in', 'is_host_login', 'is_guest_login']
    # difficulty_level is a categorical column, but we do not want to one-hot encode it, and neither should it be considered as numeric
    numeric_columns = list(
        set(dataframe.columns)
      - set(label_columns)
      - set(categorical_columns)
      - set(boolean_columns)
      - set(['difficulty_level'])
    )

    # remove attack classes: U2R
    removed_attack_classes = ['U2R']
    removed_attack_classes_index = dataframe[dataframe['attack_class'].isin(removed_attack_classes)].index
    preprocessed = dataframe.drop(index=removed_attack_classes_index).reset_index(drop=True)

    # normalize
    if normalize:
        # laod trainingset since the numeric columns should be standardized in accordance with the trainingset,
        # and we do not know if the dataframe represents the trainingset
        training = pd.read_csv('data/KDDTrain.csv', header=0, names=_HEADERS, usecols=numeric_columns)
        preprocessed[numeric_columns] = (preprocessed[numeric_columns] - training.mean(axis=0)) / training.std(axis=0)

    # split into attributes, class, attack_class
    attributes_dataframe = preprocessed.drop(columns=['class', 'attack_class'])
    class_dataframe = preprocessed['class']
    attack_class_dataframe = preprocessed['attack_class']

    # one-hot encoding
    attributes_dataframe = pd.get_dummies(attributes_dataframe, columns=categorical_columns)
    class_dataframe = pd.get_dummies(class_dataframe, columns=['class'])
    attack_class_dataframe = pd.get_dummies(attack_class_dataframe, columns=['attack_class'])

    return attributes_dataframe, class_dataframe, attack_class_dataframe

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