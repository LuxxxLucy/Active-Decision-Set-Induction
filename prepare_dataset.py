from Orange.data import Table, Domain
from Orange.data import ContinuousVariable, DiscreteVariable, StringVariable
import random
import numpy as np

import pandas as pd



'''
return dataset is a Orange Table.

I really don't like Pandas....
'''

def prepare_2d_sinusoidal_dataset(number=100):
    np.random.seed(42)
    random.seed(42)
    def myfunc(x,y):
        return 1 if np.sin(10 * x) + np.cos(4 * y)  - np.cos(3*x*y) >= 0 else 0

    np.random.seed(42)
    f_binarized = np.vectorize(myfunc)
    # generate Dataset
    delta = 0.025
    x = np.arange(0, 1, delta)
    y = np.arange(0, 2, 2*delta)
    X, Y = np.meshgrid(x, y)
    F_binarized = f_binarized(X,Y)

    X = np.stack(   [X.flatten(),Y.flatten()] ,axis=1  )
    Y = F_binarized.flatten()

    domain = Domain([ContinuousVariable("x-axis"), ContinuousVariable("y-axis")],
                    [DiscreteVariable("hit", ("no", "yes"))])
    data = Table(domain, X, Y)

    final_data_table = Table(data.domain, random.sample(data, number))

    return final_data_table



def prepare_german_dataset(filename, path_data):
    np.random.seed(42)
    # Read Dataset
    df = pd.read_csv(path_data + filename, delimiter=',')

    replace_str_function = lambda x : x.replace("_","~")
    df = df.rename(replace_str_function,axis='columns')
    from utils import data_table_from_dataframe
    data_table = data_table_from_dataframe(df,Y_column_idx=0)
    return data_table


def prepare_adult_dataset(filename, path_data):

    np.random.seed(42)
    # Read Dataset
    df = pd.read_csv(path_data + filename, delimiter=',', skipinitialspace=True)

    # Remove useless columns
    del df['fnlwgt']
    del df['education-num']

    # Remove Missing Values
    for col in df.columns:
        if '?' in df[col].unique():
            df[col][df[col] == '?'] = df[col].value_counts().index[0]

    # Read Dataset

    replace_str_function = lambda x : x.replace("_","~")
    df = df.rename(replace_str_function,axis='columns')

    from utils import data_table_from_dataframe

    data_table = data_table_from_dataframe(df,Y_column_idx=-1)

    return data_table


def prepare_compass_dataset(filename, path_data):

    # Read Dataset
    df = pd.read_csv(path_data + filename, delimiter=',', skipinitialspace=True)

    columns = ['age', 'age_cat', 'sex', 'race',  'priors_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out',
               'c_charge_degree', 'is_recid', 'is_violent_recid', 'two_year_recid', 'decile_score', 'score_text']

    df = df[columns]

    df['days_b_screening_arrest'] = np.abs(df['days_b_screening_arrest'])

    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df['length_of_stay'] = np.abs(df['length_of_stay'])

    df['length_of_stay'].fillna(df['length_of_stay'].value_counts().index[0], inplace=True)
    df['days_b_screening_arrest'].fillna(df['days_b_screening_arrest'].value_counts().index[0], inplace=True)

    df['length_of_stay'] = df['length_of_stay'].astype(int)
    df['days_b_screening_arrest'] = df['days_b_screening_arrest'].astype(int)

    def get_class(x):
        if x < 7:
            return 'Medium-Low'
        else:
            return 'High'
    df['class'] = df['decile_score'].apply(get_class)

    del df['c_jail_in']
    del df['c_jail_out']
    del df['decile_score']
    del df['score_text']

    columns = df.columns.tolist()
    columns = columns[-1:] + columns[:-1]
    df = df[columns]
    class_name = 'class'
    possible_outcomes = list(df[class_name].unique())

    type_features, features_type = recognize_features_type(df, class_name)

    discrete = ['is_recid', 'is_violent_recid', 'two_year_recid']
    discrete, continuous = set_discrete_continuous(columns, type_features, class_name, discrete=discrete,
                                                   continuous=None)

    columns_tmp = list(columns)
    columns_tmp.remove(class_name)
    idx_features = {i: col for i, col in enumerate(columns_tmp)}

    # Dataset Preparation for Scikit Alorithms
    df_le, label_encoder = label_encode(df, discrete)
    X = df_le.loc[:, df_le.columns != class_name].values
    y = df_le[class_name].values

    feature_values = calculate_feature_values(X, columns, class_name, discrete, continuous, discrete_use_probabilities=False, continuous_function_estimation=False)

    dataset = {
        'name': filename.replace('.csv', ''),
        'df': df,
        'columns': list(columns),
        'class_name': class_name,
        'possible_outcomes': possible_outcomes,
        'type_features': type_features,
        'features_type': features_type,
        'feature_values': feature_values,
        'discrete': discrete,
        'continuous': continuous,
        'idx_features': idx_features,
        'label_encoder': label_encoder,
        'X': X,
        'y': y,
    }

    return dataset
