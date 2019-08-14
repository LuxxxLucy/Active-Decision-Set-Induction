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

import operator
import math
from functools import reduce
import numpy as np
import random
from copy import deepcopy,copy
import heapq

OPERATORS = {
    # discrete, nominal variables
    # 'in': lambda x,y: all( [operator.contains(y,x_tmp) for x_tmp in x ],
    'in': lambda x,y: operator.contains(y,x),
    # continuous variables
    '<=': operator.le,
    '>=': operator.ge,
}

def rule_to_string(rule,domain,target_class_idx):
    attributes = domain.attributes
    class_var = domain.class_var
    if rule.conditions:
        cond = " AND ".join([
                            str(s.min) + " <= " +
                            attributes[s.column].name +
                            " <= " + str(s.max)
                            if attributes[s.column].is_continuous
                            else attributes[s.column].name + " is " + ",".join( [ str(attributes[s.column].values[int(v)]) for v in s.values ])
                            for s in rule.conditions
                            ])
    else:
        cond = "TRUE"
    outcome = class_var.name + "=" + class_var.values[int(target_class_idx) ]
    return "IF {} THEN {} ".format(cond, outcome)

class Condition():
    '''
    a single condition.
    It is modified based on the condition class from Orange package. (in particualr, Orange.classification.rule)
    '''
    def __init__(self, column, values=None, min_value=-math.inf,max_value=math.inf,type='categorical',domain=None):
        '''
        differs in the number of arguments as for categorical feature and continuous features
        '''
        self.type = type
        self.column = column
        if type == 'categorical':
            # for categorical features
            self.values = values[:] # value is a set of possible values
        elif type == 'continuous':
            if domain != None:

                max_limit = domain.attributes[column].max
                min_limit = domain.attributes[column].min
                min_value,max_value = sorted([min_value,max_value])
                self.min = max(min_value,min_limit) # for continuous features
                self.max = min(max_value,max_limit) # for continuous features
            else:
                self.min = min_value
                self.max = max_value
            if self.min > self.max:
                print("error!!!min and max")
                print(self.min,self.max)
                quit()
        else:
            print("critical error. Type of condition unspecified. Must be one of categorical or continuous ")


    def filter_instance(self, x):
        """
        Filter a single instance. Returns true or false
        """
        if self.type == 'categorical':
            return OPERATORS['in'](x[self.column], self.values)
        elif self.type == 'continuous':
            return OPERATORS['<='](x[self.column], self.max) and OPERATORS['>='](x[self.column], self.min)

    def filter_data(self,X):
        return self.filter_data_(X)

    def filter_data_(self,X):
    # def filter_data(self,X):
        """
        Filter several instances concurrently. Retunrs array of bools
        this is the not cached true filter_data method
        """
        if self.type == 'categorical':
            # return OPERATORS['in'](X[:,self.column],self.values)
            try:
                tmp = np.stack([np.equal(X[:,self.column], v) for v in self.values ])
                return np.any(tmp,axis=0)
            except:
                print("error in condition filter data")
                # print(self.values)
                # print(X[:5,self.column])
                # print(np.logical_or.reduce(
                # [np.equal(X[:5,self.column], v) for v in self.values ] ))
                quit()
        elif self.type == 'continuous':
            return np.logical_and(
            OPERATORS['<='](X[:,self.column], self.max),
            OPERATORS['>='](X[:,self.column], self.min)
            )

    def __eq__(self, other):
        # return self.selectors == other.selectors

        if self.type != other.type:
            # raise Exception('two selector should have the same type. The value of s1 was: {0} and the type of s2 was: {1}'.format(self.type,other.type))
            return False

        if self.type == 'categorical':
            self.values = sorted(self.values);
            other.values = sorted(other.values);
            return self.values == other.values
        elif self.type == 'continuous':
            return self.min == other.min and self.max == other.max
        else:
            raise Exception('critical error: unknown selector type {}'.format(self.type))
            return

    def __hash__(self):
        if self.type == 'categorical':
            self.values = sorted(self.values);
            return hash((self.type, self.column ,tuple(self.values)) )
        elif self.type == 'continuous':
            return hash((self.type,self.column, self.min,self.max) )
        else:
            raise Exception('critical error: unknown selector type {}'.format(self.type))
            return
        return

    def __deepcopy__(self,memodict={}):
        # copy_object = Condition()
        # copy_object.value = self.value
        if self.type == 'categorical':
            column = self.column
            values = self.values[:]
            return Condition(column=column,values=values,type='categorical')
            # return Condition(column=deepcopy(self.column,memodict),values=deepcopy(self.values,memodict),type='categorical')
        elif self.type == 'continuous':
            return  Condition(column=self.column,min_value=self.min,max_value=self.max,type='continuous')
        else:
            print("critical error. deep copy error! __deepcopy__ for Condition ")
            quit()

    def sample(self,batch_size=10):
        if self.type == 'categorical':
            synthetic_column = np.random.choice(self.values,size=batch_size)
            return synthetic_column
        elif self.type == 'continuous':
            if not isinstance(batch_size, int):
                print("strange thing happend! it is not a int:",batch_size)
            synthetic_column = np.random.uniform(low=self.min,high=self.max,size=batch_size)
            # synthetic_column = np.random.uniform(low=self.min,high=self.max,size=10)
            return synthetic_column

class Rule:
    def __init__(self,conditions,target_class_idx):
        """
        Parameters:
            feature_value_pairs : a list of (column_idx,value) pairs
            class_label: the target class
            domain: the domain
        """
        # self.conditions = conditions
        self.conditions = sorted(conditions,key=lambda x:x.column)
        for c in self.conditions:
            if c.type == 'categorical':
                c.values = sorted(c.values)
        self.target_class_idx = target_class_idx
        # self.compute_volume(domain)
        # self.domain = domain
        # if string_representation == None:
        #     self.string_representation = rule_to_string(self,domain=domain,target_class_idx=self.target_class_idx)
        # else:
        #     self.string_representation = string_representation

    def evaluate_instance(self, x):
        """
        Evaluate a single instance.

        Parameters
        ----------
        x : ndarray
            Evaluate this instance.

        Returns
        -------
        return : bool
            True, if the rule covers 'x'.
        """
        return all(condition.filter_instance(x) for condition in self.conditions)

    def evaluate_data(self, X):
        return self.evaluate_data_(X)

    def evaluate_data_(self, X):
    # def evaluate_data(self, X):
        """
        Evaluate several instances concurrently.

        Parameters
        ----------
        X : ndarray
            Evaluate this data.

        Returns
        -------
        return : ndarray, bool
            Array of evaluations.
        """
        try:
            if len(self.conditions) > 0:
                tmp = np.stack([condition.filter_data(X) for condition in self.conditions])
                # curr_covered = np.logical_and.reduce(tmp)
                curr_covered = np.all(tmp,axis=0)
            else:
                curr_covered = np.ones(X.shape[0], dtype=np.bool)
            # quit()
            return curr_covered
        except:
            # print(len(self.conditions))
            # print(self.string_representation)
            # print([len(c.values) for c in self.conditions])
            # print("*".join([str(c.values) for c in self.conditions]))
            # print([t.shape for t in tmp])
            print("error!")
            quit()

    def __len__(self):
        return len(self.conditions)

    def __hash__(self):
        return hash(self.string_representation)

    def __eq__(self, other):
        return self.string_representation == other.string_representation

    def __sub__( self, other ):
        # conditions = deepcopy(self.conditions)
        conditions = []
        for i,(c,tmp_c) in enumerate(zip(self.conditions,other.conditions)):
            conditions.append(c-tmp_c)
        # for i,c in enumerate(self.conditions):
        #     for tmp_c in other.conditions:
        #         conditions[i] = c - tmp_c
        new_rule = deepcopy(self)
        new_rule.conditions = conditions
        return new_rule

    def __lt__(self,other):
        # return hash(self) < hash(other)
        return self.string_representation <= other.string_representation

    def __deepcopy__(self,memodict={}):
        return Rule(conditions=[ deepcopy(c,memodict) for c in self.conditions],target_class_idx=self.target_class_idx)

    def add_constraint(self,feature,threshold,domain,mode="left_child"):
        if domain.attributes[feature].is_continuous:
            type = "continuous"
            if feature in [c.column for c in self.conditions]:
                condition = [ c for c in self.conditions if c.column==feature]
                assert len(condition)==1,"Strange thing! One condition should only appear one time"
                condition = condition[0]
                if mode == "left_child":
                    condition.max = threshold
                else:
                    condition.min = threshold
            else:
                if mode == "left_child":
                    new_condition = Condition(column=feature,max_value=threshold,domain=domain,type='continuous')
                else:
                    new_condition = Condition(column=feature,min_value=threshold,domain=domain,type='continuous')
                self.conditions.append(new_condition)
                self.conditions = sorted(self.conditions,key= lambda x:x.column)
        else:
            type = "categorical"
            if feature in [c.column for c in self.conditions]:
                condition = [ c for c in self.conditions if c.column==feature]
                assert len(condition)==1,"Strange thing! One condition should only appear one time"
                condition = condition[0]
                all_possoble_values = condition.values
                if mode == "left_child":
                    new_values = [ v for v in all_possoble_values if v <= threshold]
                    condition.values = new_values
                else:
                    new_values = [ v for v in all_possoble_values if v > threshold]
                    condition.values = new_values
            else:
                all_possoble_values = range(len(domain.attributes[feature].values))
                if mode == "left_child":
                    new_values = [ v for v in all_possoble_values if v <= threshold]
                    new_condition = Condition(column=feature,values=new_values,type='categorical')
                else:
                    new_values = [ v for v in all_possoble_values if v > threshold]
                    new_condition = Condition(column=feature,values=new_values,type='categorical')
                self.conditions.append(new_condition)

            self.conditions = sorted(self.conditions,key= lambda x:x.column)


    def set_target_class_idx(self,idx,domain):
        self.target_class_idx=idx
        self.string_representation = rule_to_string(self,domain=domain,target_class_idx=self.target_class_idx)
