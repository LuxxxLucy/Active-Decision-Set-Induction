import numpy as np

def label_with_blackbox(data_table,blackbox):
    from Orange.data import Table
    return Table.from_numpy(X=data_table.X,Y=blackbox(data_table.X),domain=data_table.domain)

def compute_metrics(rules,domain):
    '''compute various metrics for rules'''

    print("number of rules",len(rules))
    print("ave number of conditions" , sum([ len(r.conditions) for r in rules]) / len(rules) )
    print("max number of conditions" , max([ len(r.conditions) for r in rules]) )
    print("used features", len(set(  [ c.column for r in rules for c in r.conditions]  ) )  )


def ruleset_predict(ruleset,X):
    # curr_covered_or_not = np.zeros(X.shape[0], dtype=np.bool)
    # for r in ruleset:
    #     curr_covered_or_not |= r.evaluate_data(X)
    if len(ruleset) > 0:
        curr_covered_or_not = np.logical_or.reduce( [r.evaluate_data_(X) for r in ruleset] )
    else:
        curr_covered_or_not = np.zeros(X.shape[0], dtype=np.bool)
    return curr_covered_or_not

def bds_ruleset_predict(ruleset,X,domain):
    import operator

    curr_covered_or_not = np.zeros(X.shape[0], dtype=np.bool)
    for r in ruleset:

        curr_covered = np.ones(X.shape[0], dtype=bool)
        for condition in r.selectors:
            if condition.type == 'categorical':
                tmp = np.logical_or.reduce(
                [np.equal(X[:,condition.column], domain.attributes[condition.column].values.index(v)  ) for v in condition.values ]
                )
            elif condition.type == 'continuous':
                tmp = np.logical_and(
                operator.lt(X[:,condition.column], condition.max),
                operator.gt(X[:,condition.column], condition.min)
                )

            curr_covered &= tmp

        curr_covered_or_not |=  curr_covered
    return curr_covered_or_not

def data_table_from_dataframe(dataframe,Y_column_idx=0):
    import numpy as np
    from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
    def series2descriptor(d):
        if d.dtype is np.dtype("float") or d.dtype is np.dtype("int"):
            return ContinuousVariable(str(d.name))
        else:
            t = d.unique()
            t.sort()
            return DiscreteVariable(str(d.name), list(t.astype("str")))

    def df2domain(df):
        # domain of X
        X_columns = [it for it in range(len(df.columns)) if it!= Y_column_idx ]
        X_feature_list = [series2descriptor(df.iloc[:,col]) for col in X_columns   ]

        # domain of Y
        Y_series = df.iloc[:,Y_column_idx]
        tmp = Y_series.unique();tmp.sort()
        Y_feature = [ DiscreteVariable(str(Y_series.name), list(tmp.astype("str")))  ]

        return Domain(X_feature_list,Y_feature)

    def df2table(df):
        domain = df2domain(df)
        # print(domain.attributes)
        Y_values = df.iloc[:,Y_column_idx].astype("str").values
        X_columns = [it for it in range(len(df.columns)) if it!= Y_column_idx ]
        X_values = df.iloc[:,X_columns].values
        X=[];Y=[]
        for index, (row_x,row_y) in enumerate(zip(X_values,Y_values) ):
            x = []
            for col,it in enumerate(row_x):
                if domain.attributes[col].is_discrete:
                    x.append( domain.attributes[col].values.index(it) )
                else:
                    x.append( it )
            X.append(x)
            Y.append( domain.class_var.values.index(row_y) )
        X = np.array(X)
        Y = np.array(Y)

        return Table(domain, X,Y)
    if Y_column_idx < 0:
        Y_column_idx = len(dataframe.columns) + Y_column_idx
    data_table = df2table(dataframe)

    return data_table,dataframe

def encoder_from_datatable(data_table):
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import make_column_transformer
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    # encoder = collections.namedtuple('random_name',
    #                                       ['fit_transform'])(lambda x: x)
    categorical_features = [a.name for a in data_table.domain.attributes if a.is_discrete]
    categorical_features_idx = [i for i,a in enumerate(data_table.domain.attributes) if a.is_discrete]
    numerical_features = [a.name for a in data_table.domain.attributes if a.is_continuous]
    numerical_features_idx = [i for i,a in enumerate(data_table.domain.attributes) if a.is_continuous]
    preprocess_encoder = make_column_transformer(
                            ( make_pipeline(SimpleImputer(), StandardScaler()), numerical_features_idx  ),
                            ( OneHotEncoder(categories='auto'),categorical_features_idx)
                            )
    preprocess_encoder.fit(data_table.X)
    return preprocess_encoder


def rule_to_string_BRS_compat(rule,domain,target_class_idx):
    attributes = domain.attributes
    class_var = domain.class_var
    if rule.selectors:
        cond = " AND ".join([
                            str(s.min) + " <= " + attributes[s.column].name + " <= " + str(s.max) if attributes[s.column].is_continuous else attributes[s.column].name + " in " + ",".join ( [ cat_value for cat_value in s.values ] )  for s in rule.selectors ]
                            )
        # for s in rule.selectors:
        #     print(s.column)
    else:
        cond = "TRUE"
    outcome = class_var.name + "=" + class_var.values[target_class_idx]
    return "IF {} THEN {} ".format(cond, outcome)

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
    outcome = class_var.name + "=" + class_var.values[target_class_idx]
    return "IF {} THEN {} ".format(cond, outcome)

def uniform_enlarge_dataset(data_table,black_box,sampling_rate=1,random_seed=42):

    from Orange.data import Table
    if sampling_rate == 0:
        # print("sampling rate zero")
        return Table.from_numpy(data_table.domain,X=data_table.X,Y=data_table.Y)

    from core import extend_rule,uniform_sampling,core_init
    from structure import Rule
    from utils import  rule_to_string

    import numpy as np
    core_init(random_seed,data_table)
    domain = data_table.domain
    # for continuous variabels, we compute max and min
    for feature in domain.attributes:
        if feature.is_continuous:
            feature.max = max([ d[feature] for d in data_table ]).value
            feature.min = min([ d[feature] for d in data_table ]).value
    empty_rule = Rule(conditions=[],domain=domain,target_class_idx=1 )
    empty_rule = extend_rule(empty_rule,data_table.domain)

    sampled_population = int(sampling_rate*data_table.X.shape[0])
    synthetic_instances_X = uniform_sampling(empty_rule,population_size=sampled_population)
    synthetic_instances_Y = black_box(synthetic_instances_X)
    print(synthetic_instances_Y)
    X_new = np.concatenate((data_table.X,synthetic_instances_X) )
    Y_new = np.concatenate((data_table.Y,synthetic_instances_Y) )

    new_data_table = Table.from_numpy(data_table.domain,X=X_new,Y=Y_new)
    return new_data_table

def estimated_enlarge_dataset(data_table,black_box,sampling_rate=1,random_seed=42):
    '''
    the following code is from RuleMatrix https://github.com/rulematrix/rule-matrix-py
    It first estimate P(X) and then sample from it
    '''
    from Orange.data import Table
    if sampling_rate == 0:
        # print("sampling rate zero: estimated")
        return Table.from_numpy(data_table.domain,X=data_table.X,Y=data_table.Y)

    from core import extend_rule,uniform_sampling
    from structure import Rule
    from utils import  rule_to_string
    import numpy as np

    from typing import Callable, List, Optional, Union
    from collections import defaultdict, namedtuple

    from numpy.random import multivariate_normal
    from scipy import stats

    ArrayLike = Union[List[Union[np.ndarray, float]], np.ndarray]

    INTEGER = 'integer'
    CONTINUOUS = 'continuous'
    CATEGORICAL = 'categorical'

    data_type = {INTEGER, CONTINUOUS, CATEGORICAL}


    class IntegerConstraint(namedtuple('IntegerConstraint', ['range_'])):

        def regularize(self, arr: np.ndarray):
            arr = np.round(arr)
            if self.range_ is not None:
                assert len(arr.shape) == 1
                arr[arr > self.range_[1]] = self.range_[1]
                arr[arr < self.range_[0]] = self.range_[0]
            return arr


    CategoricalConstraint = namedtuple('CategoricalConstraint', ['categories'])


    class ContinuousConstraint(namedtuple('ContinuousConstraint', ['range_'])):

        def regularize(self, arr: np.ndarray):
            if self.range_ is not None:
                arr = arr.copy()
                assert len(arr.shape) == 1
                arr[arr > self.range_[1]] = self.range_[1]
                arr[arr < self.range_[0]] = self.range_[0]
            return arr


    def create_constraint(feature_type, **kwargs):
        if feature_type == INTEGER:
            return IntegerConstraint(kwargs['range_'])
        elif feature_type == CONTINUOUS:
            return ContinuousConstraint(kwargs['range_'])
        elif feature_type == CATEGORICAL:
            return CategoricalConstraint(None)
        else:
            raise ValueError("Unknown feature_type {}".format(feature_type))


    def create_constraints(n_features, is_continuous: np.ndarray=None, is_categorical: np.ndarray=None,
                           is_integer: np.ndarray=None, ranges=None):
        is_continuous, is_categorical, is_integer = \
            check_input_constraints(n_features, is_continuous, is_categorical, is_integer)
        constraints = []
        for i in range(len(is_categorical)):
            feature_type = CATEGORICAL if is_categorical[i] else CONTINUOUS if is_continuous[i] else INTEGER
            constraints.append(create_constraint(feature_type, range_=ranges[i]))
        return constraints


    def check_input_constraints(n_features, is_continuous=None, is_categorical=None, is_integer=None):
        if is_integer is None:
            is_integer = np.zeros((n_features,), dtype=np.bool)
        else:
            is_integer = np.array(is_integer, dtype=np.bool)
            if is_integer.shape != (n_features,):
                raise ValueError("is_integer should be None or an array of bool mask!")
        if is_categorical is None:
            is_categorical = np.zeros((n_features,), dtype=np.bool)
        else:
            is_categorical = np.array(is_categorical, dtype=np.bool)
            if is_categorical.shape != (n_features,):
                raise ValueError("is_categorical should be None or an array of bool mask!")
        if is_continuous is None:
            is_continuous = np.logical_not(np.logical_and(is_integer, is_categorical))
        else:
            is_continuous = np.array(is_continuous, dtype=np.bool)
            if is_continuous.shape != (n_features,):
                raise ValueError("is_continuous should be None or an array of bool mask!")

        if np.any(np.logical_and(is_categorical, np.logical_and(is_integer, is_continuous))) \
                or not np.all(np.logical_or(is_categorical, np.logical_or(is_integer, is_continuous))):
            raise ValueError("is_integer, is_categorical, and is_continuous "
                             "should be exclusive and collectively exhaustive!")

        return is_continuous, is_categorical, is_integer


    def gaussian_mixture(means: np.ndarray,
                         cov: np.ndarray,
                         weights: Optional[np.ndarray] = None,
                         random_state=None
                         ) -> Callable[[int], np.ndarray]:
        if weights is None:
            weights = np.empty(len(means), dtype=np.float32)
            weights.fill(1 / len(means))
        n_features = len(means[0])

        def sample(n: int) -> np.ndarray:
            if random_state:
                norm = random_state.multivariate_normal(np.zeros((n_features,), dtype=np.float), cov, n)
                indices = random_state.choice(len(means), size=n, p=weights)
            else:
                norm = multivariate_normal(np.zeros((n_features,), dtype=np.float), cov, n)
                indices = np.random.choice(len(means), size=n, p=weights)
            return norm + means[indices]

        return sample


    def scotts_factor(n, d):
        return n ** (-1. / (d + 4))


    def create_sampler(instances: np.ndarray, is_continuous=None, is_categorical=None, is_integer=None, ranges=None,
                       cov_factor=1.0, seed=None, verbose=False) -> Callable[[int], np.ndarray]:
        """
        We treat the sampling of categorical values as a multivariate categorical distribution.
        We sample categorical values first, then sample the continuous and integer variables
        using the conditional distribution w.r.t. to the categorical vector.
        Note: each category feature only support at most **128** number of choices
        :param instances: np.ndarray
            the data that we used for building estimation and sampling new data.
            A 2D array of shape [n_instances, n_features]
        :param is_continuous: np.ndarray (default=None)
            A bool mask array indicating whether each feature is continuous or not.
            If all three masks are all None, then by default all features are continuous.
        :param is_categorical: np.ndarray (default=None)
            A bool mask array indicating whether each feature is categorical.
        :param is_integer: np.ndarray (default=None)
            A bool mask array indicating whether each feature is integer.
        :param ranges: List[Optional[(float, float)]]
            A list of (min, max) or None, indicating the ranges of each feature.
        :param cov_factor: float (default=1.0)
            A multiplier that scales the covariance matrix.
        :param seed: Random seed for the sampling
        :param verbose: a flag for debugging output
        :return: a function handle (n: int) -> np.ndarray that creates samples from the estimated distribution
        """

        # Check the shape of input
        instances = np.array(instances)
        assert len(instances.shape) == 2
        n_features = instances.shape[1]
        n_samples = len(instances)

        is_continuous, is_categorical, is_integer = check_input_constraints(
            n_features, is_continuous, is_categorical, is_integer)
        is_numeric = np.logical_or(is_integer, is_continuous)
        if ranges is None:
            ranges = [None] * n_features

        # Build up constraints
        constraints = create_constraints(n_features, is_continuous, is_categorical, is_integer, ranges)

        # Create RandomState
        random_state = np.random.RandomState(seed=seed)

        def _build_cache():
            categoricals = instances[:, is_categorical].astype(np.int8)

            categorical_samples = defaultdict(list)
            for i in range(n_samples):
                key = bytes(categoricals[i, :])
                categorical_samples[key].append(instances[i])

            keys = []
            probs = []
            key2instances = []
            for key, value in categorical_samples.items():
                keys.append(key)
                probs.append(len(value) / n_samples)
                key2instances.append(np.array(value))
            if verbose:
                print("# of categories:", len(keys))
                print("Distribution of # of instances per categories:")
                hists, bins = np.histogram(probs * n_samples, 5)
                print("hists:", hists.tolist())
                print("bins:", bins.tolist())
            return keys, probs, key2instances

        cat_keys, cat_probs, cat2instances = _build_cache()

        # Try stats.gaussian_kde
        continuous_data = instances[:, is_numeric]
        n_numeric = np.sum(is_numeric)

        # Estimate the covariance matrix for kde
        if n_numeric != 0:
            glb_kde = stats.gaussian_kde(continuous_data.T, 'silverman')
            cov = cov_factor * glb_kde.covariance
        else:
            cov = []

        # The sampling function
        def sample(n: int) -> np.ndarray:
            samples = []
            # Sample categorical features by multinomial
            sample_nums = random_state.multinomial(n, cat_probs)
            for idx, num in enumerate(sample_nums):
                if num == 0:
                    continue
                sample_buffer = np.empty((num, n_features), dtype=np.float)

                # Sample continuous part
                if n_numeric != 0:
                    sample_buffer[:, is_numeric] = gaussian_mixture(cat2instances[idx][:, is_numeric], cov,
                                                                    random_state=random_state)(num)
                # Fill the categorical part
                categorical_part = np.frombuffer(cat_keys[idx], dtype=np.int8)
                sample_buffer[:, is_categorical] = np.tile(categorical_part, (num, 1)).astype(np.float)

                samples.append(sample_buffer)
            sample_mat = np.vstack(samples)

            # regularize numeric part
            for i, constraint in enumerate(constraints):
                if isinstance(constraint, IntegerConstraint) or isinstance(constraint, ContinuousConstraint):
                    sample_mat[:, i] = constraint.regularize(sample_mat[:, i])

            return sample_mat

        return sample

    from typing import Optional
    domain = data_table.domain
    # for continuous variabels, we compute max and min
    ranges = None
    # for feature in domain.attributes:
    #     if feature.is_continuous:
    #         feature.max = max([ d[feature] for d in data_table ]).value
    #         feature.min = min([ d[feature] for d in data_table ]).value
    #         ranges.append(Optional[(feature.min,feature.max)])
    #     else:
    #         ranges.append(None)


    sampled_population = int(sampling_rate*data_table.X.shape[0])

    # synthetic_instances_X = uniform_sampling(empty_rule,population_size=20000)
    is_continuous_mask = np.array([ 1 if a.is_continuous else 0 for a in domain.attributes])
    is_categorical_mask = np.array([ 1 if a.is_discrete else 0 for a in domain.attributes])
    is_int_mask = np.array([ 0 for a in domain.attributes])
    cov_factor=1.0
    sampler = create_sampler(data_table.X, is_continuous_mask, is_categorical_mask, is_int_mask,
                                                ranges, cov_factor, seed=random_seed)
    synthetic_instances_X  = sampler(sampled_population)
    synthetic_instances_Y = black_box(synthetic_instances_X)
    X_new = np.concatenate((data_table.X,synthetic_instances_X) )
    Y_new = np.concatenate((data_table.Y,synthetic_instances_Y) )

    new_data_table = Table.from_numpy(data_table.domain,X=X_new,Y=Y_new)
    return new_data_table
