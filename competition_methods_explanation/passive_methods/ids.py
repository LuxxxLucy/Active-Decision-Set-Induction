# import general package
import numpy as np
import random
from collections import namedtuple
import operator

# import particular package we utilize
import Orange

# import local package
from .IDS.IDS_smooth_local import *
from .IDS.utils import rules_convert

def explain_tabular(dataset, blackbox, target_class_idx=1, pre_label=True, random_seed=42):
    '''
    Input Params:
    1. dataset: a Orange data table
    3. blackbox: a blackbox predict function, such as `c.predict` where c is a scikit-classifier
    4. target_class
    ---
    Output:
    A decision set.
    '''
    np.random.seed(random_seed)

    if pre_label == False:
        # re-labelled the data using the blackbox, otherwise assuming the labels provided is labeled by the classifier, (instead of the groundtruth label)
        labels = blackbox(dataset.X)
        dataset = Orange.data.Table(dataset.domain, dataset.X, labels)

    # fit the explainer to the data
    # explainer = IDS(dataset, blackbox)
    # rule_set = explainer.fit(dataset.domain,dataset.X,dataset.Y,target_class=target_class)


    # df = pd.read_csv('titanic_train.tab',' ', header=None, names=['Passenger_Cat', 'Age_Cat', 'Gender'])
    # df1 = pd.read_csv('titanic_train.Y', ' ', header=None, names=['Died', 'Survived'])
    # Y = list(df1['Died'].values)
    # df1.head()

    import Orange
    from Orange.data.pandas_compat import table_to_frame
    import pandas as pd
    if all([ a.is_discrete for a in dataset.domain.attributes]) == True:
        disc_data_table = dataset
    else:
        print("discre")
        disc = Orange.preprocess.Discretize()
        disc.method = Orange.preprocess.discretize.EqualFreq(n=5)
        # disc.method = Orange.preprocess.discretize.EntropyMDL(force=True)
        disc_data_table = disc(dataset)
        # df = table_to_frame(disc_data_table)
        # Y = pd.DataFrame(disc_data_table.Y,columns=[disc_data_table.domain.class_var.name],dtype='int32')

    assert all([ a.is_discrete for a in disc_data_table.domain.attributes]), " is not pre-discretized!"
    # disc_data_table = dataset
    Y = disc_data_table.Y
    df = table_to_frame(disc_data_table)
    df.drop(df.columns[-1],axis = 1, inplace = True)


    print("start Apriori")
    itemsets = run_apriori(df, 0.05)
    # itemsets = run_apriori(df, 0.5)
    print("finish Apriori. Converting itemset")
    list_of_rules = createrules(itemsets, list(set(Y)))
    print("Pre-mined okay. all pre-mined rules of",len(list_of_rules))
    # print("----------------------")
    # for r in list_of_rules:
    #     r.print_rule()

    # lambda_array = [1.0]*7     # use separate hyperparamter search routine
    lambda_array = [0.5,1.0,1.0,1.0,1.0,1.5,1.0]     # use separate hyperparamter search routine
    s1 = smooth_local_search(list_of_rules, df, Y, lambda_array, 0.33, 0.33)
    s2 = smooth_local_search(list_of_rules, df, Y, lambda_array, 0.33, -1.0)
    f1 = func_evaluation(s1, list_of_rules, df, Y, lambda_array)
    f2 = func_evaluation(s2, list_of_rules, df, Y, lambda_array)
    if f1 > f2:
        print("The Solution Set is: "+str(s1))
        rule_set = [ list_of_rules[idx] for idx in s1]
    else:
        print("The Solution Set is: "+str(s2))
        rule_set = [ list_of_rules[idx] for idx in s2]
        print(rule_set)

    # convert the rule representation
    rule_set = rules_convert(rule_set,dataset, target_class_idx=target_class_idx)
    return rule_set

def compute_metrics_ids(rules):
    print("number of rules",len(rules))
    if len(rules) != 0:
        print("ave number of conditions" , sum([ len(r.conditions) for r in rules]) / len(rules) )
        print("max number of conditions" , max([ len(r.conditions) for r in rules]) )
        print("used features", len(set(  [ c.column for r in rules for c in r.conditions]  ) )  )
    else:
        print("ave number of conditions" , 0 )
        print("max number of conditions" , 0 )
        print("used features", 0 )
    return
