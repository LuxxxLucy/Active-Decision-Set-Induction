from Orange.data import Table, Domain
from Orange.data import ContinuousVariable, DiscreteVariable, StringVariable
import random
import numpy as np

import pandas as pd



'''
return dataset is a Orange Table.

I really don't like Pandas....
'''
from Orange.data import Table
from sklearn.model_selection import train_test_split



def train_test_split_data(data_table):
    X_train,X_test,y_train,y_test = train_test_split(data_table.X,data_table.Y,test_size=0.33,random_state=42)


    train_data_table = Table.from_numpy(X=X_train,Y=y_train,domain=data_table.domain)

    test_data_table = Table.from_numpy(X=X_test,Y=y_test,domain=data_table.domain)

    return train_data_table,test_data_table

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

    data_table = Table(data.domain, random.sample(data, number))
    train_table, test_table = train_test_split_data(data_table)

    return train_table,test_table

def prepare_compas_dataset(train_filename="compas_train.csv",test_filename="compas_test.csv",path_data="./datasets/compas/"):
    np.random.seed(42)
    df = pd.read_csv(path_data + train_filename, delimiter=',')
    from utils import data_table_from_dataframe
    train_data_table,df = data_table_from_dataframe(df,Y_column_idx=-1)

    df = pd.read_csv(path_data + test_filename, delimiter=',')
    from utils import data_table_from_dataframe
    test_data_table,df = data_table_from_dataframe(df,Y_column_idx=-1)

    return train_data_table,test_data_table

def prepare_diabetes_dataset(filename="diabetic_data.csv",path_data="./datasets/diabetes/"):
    np.random.seed(42)
    df = pd.read_csv(path_data + filename, delimiter=',')

    # for col in df.columns:
    # if '?' in df[col].unique():
    #     df[col][df[col] == '?'] = df[col].value_counts()[0]

    # remove information-less ids
    df = df.drop(df.columns[[0, 1]], axis=1)
    continuous_columns = [7,10,11,12,13,14,15,19]
    # for i,c in enumerate(df.columns)
    #     if i in continuous_columns:
    #         df[c].astype("float64")
    #     else:
    #         df[c].astype("str")

    from utils import data_table_from_dataframe
    train_data_table,df = data_table_from_dataframe(df,Y_column_idx=-1)

    # df = pd.read_csv(path_data + test_filename, delimiter=',')
    # from utils import data_table_from_dataframe
    # test_data_table,df = data_table_from_dataframe(df,Y_column_idx=-1)
    return train_data_table,None




def prepare_german_dataset(filename="german_credit.csv", path_data="./datasets/german/"):
    np.random.seed(42)
    # Read Dataset
    df = pd.read_csv(path_data + filename, delimiter=',')

    replace_str_function = lambda x : x.replace("_","~")
    df = df.rename(replace_str_function,axis='columns')
    from utils import data_table_from_dataframe
    data_table,df = data_table_from_dataframe(df,Y_column_idx=0)
    train_table, test_table = train_test_split_data(data_table)
    return train_table,test_table


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

    data_table,df = data_table_from_dataframe(df,Y_column_idx=-1)

    return data_table,df
