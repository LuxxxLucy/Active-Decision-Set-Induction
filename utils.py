import re
from structure import Condition


def add_condition(rule,rule_idx,attribute_idx,value,domain):
    '''
    add a condition into a rule
    '''
    attribute = domain.attributes[attribute_idx]
    col_idx = attribute_idx
    if attribute.is_discrete:
        s = Condition(column=col_idx,values=[value],type='categorical')
        rule.conditions.append(s)
    elif attribute.is_continuous:
        c = re.split( "- | ",value)
        max_value = domain.attributes[attribute_idx].max
        min_value = domain.attributes[attribute_idx].min
        if len(c) == 2:
            if c[0] in ['<=','≤','<']:
                s = Condition(column=col_idx,max_value=float(c[1]),type='continuous',possible_range=(min_value,max_value))
            else:
                s = Condition(column=col_idx,min_value=float(c[1]),type='continuous',possible_range=(min_value,max_value))
            rule.conditions.append(s)
        if len(c) == 3:
            s = Condition(column=col_idx,min_value=float(c[0]),max_value=float(c[2]),type='continuous',possible_range=(min_value,max_value))
            rule.conditions.append(s)
    else:
        raise ValueError('not possible, must be discrete or continuous:',attribute)

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

def itemsets_to_rules(itemsets,domain,decoder_item_to_feature,target_class):


    from structure import Condition,Rule
    import re
    rule_set = []
    col_names = [it.name for it in domain.attributes]
    possible_range = [ (it.min,it.max) if it.is_continuous else (0,0) for it in domain.attributes   ]
    for itemset,_ in itemsets:
        previous_cols =[]
        conditions=[]
        for item in itemset:
            name,string_conditions = decoder_item_to_feature[item]
            col_idx = col_names.index(name)
            if col_idx not in previous_cols:
                if domain.attributes[col_idx].is_continuous:
                    c = re.split( "- | ",string_conditions)
                    if len(c) == 2:
                        if c[0] in ['<=','≤','<']:
                            s = Condition(column=col_idx,max_value=float(c[1]),type='continuous',possible_range=possible_range[col_idx])
                        else:
                            s = Condition(column=col_idx,min_value=float(c[1]),type='continuous',possible_range=possible_range[col_idx])
                        conditions.append(s)
                    if len(c) == 3:
                        s = Condition(column=col_idx,min_value=float(c[0]),max_value=float(c[2]),type='continuous',possible_range=possible_range[col_idx])
                        conditions.append(s)
                else:
                    # categorical
                    s = Condition(column=col_idx,values=[string_conditions],type='categorical')
                    conditions.append(s)
            else:
                s = conditions[ previous_cols.index(col_idx) ]
                if domain.attributes[col_idx].is_continuous:
                    c = re.split( "- | ",string_conditions)
                    if len(c) == 2:
                        if c[0] in ['<=','≤','<']:
                            s_tmp = Condition(column=col_idx,max_value=float(c[1]),type='continuous',possible_range=possible_range[col_idx])
                        else:
                            s_tmp = Condition(column=col_idx,min_value=float(c[1]),type='continuous',possible_range=possible_range[col_idx])
                        s.max = max(s.max,s_tmp.max)
                        s.min = min(s.min,s_tmp.min)
                    if len(c) == 3:
                        s.max = max(s.max,float(c[2]))
                        s.min = min(s.min,float(c[0]))
                else:
                    # categorical
                    s.values.append(string_conditions)
            previous_cols.append(col_idx)

        # merge, for example, merge "1<x<2" and "2<x<3" into "1<x<3"
        rule = Rule(conditions=conditions,domain=domain,target_class=target_class)
        rule_set.append(rule)

    return rule_set

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
