import copy
import sklearn
import numpy as np
import lime
import lime.lime_tabular
# import string
import os
import sys
import pandas
import collections

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class Bunch(object):
    """bla"""
    def __init__(self, adict):
        self.__dict__.update(adict)

def load_dataset_dataframe(data_frame,target_class="class",feature_to_use=None,categorical_features=None, balance=False,discretize=False):

    target_idx = data_frame.columns.get_loc(target_class)
    # feature_names = data_frame.drop([target_class],axis=1).columns.values
    feature_names = data_frame.columns.values
    if feature_to_use == None:
        feature_to_use = [ it for it in range(len(data_frame.columns.values)) if it !=target_idx  ]

    if categorical_features == None:
        categorical_features_name =  data_frame.drop([target_class],axis=1).select_dtypes(exclude=['int', 'float']).columns.values
        categorical_features = [data_frame.columns.get_loc(c) for c in categorical_features_name if c in data_frame]

    dataset = preprocess_dataset(
        data_frame, target_idx,
        feature_names=feature_names,
        features_to_use=feature_to_use,
        categorical_features=categorical_features, discretize=discretize,
        balance=balance)
    return dataset

def preprocess_dataset(data_frame, target_idx,
                     feature_names=None, categorical_features=None,
                     features_to_use=None,
                     discretize=False, balance=False, fill_na='-1', filter_fn=None, skip_first=False):
    """if not feature names, takes 1st line as feature names
    if not features_to_use, use all except for target
    if not categorical_features, consider everything < 20 as categorical"""
    data = data_frame
    ret = Bunch({})
    if feature_names is None:
        feature_names = list(data[0])
        data = data[1:]
    else:
        feature_names = copy.deepcopy(feature_names)
    if skip_first:
        data = data[1:]
    if filter_fn is not None:
        data = filter_fn(data)
    labels = data.iloc[:, target_idx].values
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(labels)
    ret.labels = le.transform(labels)
    labels = ret.labels
    ret.class_names = list(le.classes_)
    ret.target_names = ret.class_names
    ret.class_target = feature_names[target_idx]
    if features_to_use is not None:
        data = data.iloc[:, features_to_use].values
        feature_names = ([x for i, x in enumerate(feature_names)
                          if i in features_to_use])
        if categorical_features is not None:
            categorical_features = ([features_to_use.index(x)
                                     for x in categorical_features])
    else:
        data = np.delete(data, target_idx, 1)
        feature_names.pop(target_idx)
        if categorical_features:
            categorical_features = ([x if x < target_idx else x - 1
                                     for x in categorical_features])

    if categorical_features is None:
        categorical_features = []
        for f in range(data.shape[1]):
            if len(np.unique(data[:, f])) < 20:
                categorical_features.append(f)
    categorical_names = {}
    for feature in categorical_features:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(data[:, feature])
        data[:, feature] = le.transform(data[:, feature])
        categorical_names[feature] = le.classes_
    data = data.astype(float)
    ordinal_features = []
    if discretize:
        disc = lime.lime_tabular.QuartileDiscretizer(data,
                                                     categorical_features,
                                                     feature_names)
        data = disc.discretize(data)
        ordinal_features = [x for x in range(data.shape[1])
                            if x not in categorical_features]
        categorical_features = range(data.shape[1])
        categorical_names.update(disc.names)
    for x in categorical_names:
        categorical_names[x] = [y.decode() if type(y) == np.bytes_ else y for y in categorical_names[x]]
    ret.ordinal_features = ordinal_features
    ret.categorical_features = categorical_features
    ret.categorical_names = categorical_names
    ret.feature_names = feature_names
    np.random.seed(1)
    if balance:
        idxs = np.array([], dtype='int')
        min_labels = np.min(np.bincount(labels))
        for label in np.unique(labels):
            idx = np.random.choice(np.where(labels == label)[0], min_labels)
            idxs = np.hstack((idxs, idx))
        data = data[idxs]
        labels = labels[idxs]
        ret.data = data
        ret.labels = labels

    ret.target = labels

    splits = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                  test_size=.2,
                                                  random_state=1)
    train_idx, test_idx = [x for x in splits.split(data)][0]
    ret.train = data[train_idx]
    ret.labels_train = ret.labels[train_idx]
    cv_splits = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                     test_size=.5,
                                                     random_state=1)
    cv_idx, ntest_idx = [x for x in cv_splits.split(test_idx)][0]
    cv_idx = test_idx[cv_idx]
    test_idx = test_idx[ntest_idx]

    ret.validation = data[cv_idx]
    ret.labels_validation = ret.labels[cv_idx]
    ret.test = data[test_idx]
    ret.labels_test = ret.labels[test_idx]
    ret.test_idx = test_idx
    ret.validation_idx = cv_idx
    ret.train_idx = train_idx
    ret.data = data
    return ret

def encoder_from_dataset(dataset):
    encoder = collections.namedtuple('random_name',
                                          ['fit_transform'])(lambda x: x)
    if dataset.categorical_names:
        # TODO: Check if this n_values is correct!!
        cat_names = sorted(dataset.categorical_names.keys())
        n_values = [len(dataset.categorical_names[i]) for i in cat_names]

        # this is deprecated
        # encoder = sklearn.preprocessing.OneHotEncoder(
        #     categorical_features=cat_names,
        #     n_values=n_values)
        # encoder.fit(dataset.data)
        encoder = ColumnTransformer(
            [('one_hot_encoder', OneHotEncoder(categories='auto'), cat_names  )], remainder='passthrough')
        encoder.fit_transform(dataset.data)


    return encoder


def discretizer_from_dataset(dataset,discretizer = 'quartile'):
    categorical_features = sorted(dataset.categorical_names.keys())

    if discretizer == 'quartile':
        disc = lime.lime_tabular.QuartileDiscretizer(dataset.data,
                                                     categorical_features,
                                                     dataset.feature_names)
    elif discretizer == 'decile':
        disc = lime.lime_tabular.DecileDiscretizer(dataset.data,
                                                 categorical_features,
                                                 dataset.feature_names)
    else:
        raise ValueError('Discretizer must be quartile or decile')

    return disc
