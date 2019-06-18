


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


def rule_to_string(rule,domain,target_class_idx):
    attributes = domain.attributes
    class_var = domain.class_var
    if rule.conditions:
        cond = " AND ".join([
                            str(s.min) + " <= " +
                            attributes[s.column].name +
                            " <= " + str(s.max)
                            if attributes[s.column].is_continuous
                            else attributes[s.column].name + " in " + str(attributes[s.column].values[int(s.values)])
                            for s in rule.conditions
                            ])
    else:
        cond = "TRUE"
    outcome = class_var.name + "=" + class_var.values[target_class_idx]
    return "IF {} THEN {} ".format(cond, outcome)
