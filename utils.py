
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


def rule_to_string_BDS_compat(rule,domain,target_class_idx):
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
