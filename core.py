# import general package function
import numpy as np
import heapq
import math
from copy import deepcopy
import time
import random
random.seed(42)
# import local function
# from objective import objective

import warnings
warnings.filterwarnings("ignore")

import sklearn
from sklearn.metrics.pairwise import euclidean_distances as distance_function
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder,Normalizer

from Orange.data import Table
from Orange.distance import Euclidean

import sys
this = sys.modules[__name__]
# this.distance_model = None
this.transformer = None

def core_init(seed,data_table):
    random.seed(seed)
    np.random.seed(seed)
    # this.distance_model = Euclidean(normalize=True).fit(data_table)
    print("init transformer okay!")
    domain = data_table.domain
    categorical_features_idx = [i for i,a in enumerate(domain.attributes) if a.is_discrete]
    continuous_features_idx = [i for i,a in enumerate(domain.attributes) if a.is_continuous]
    this.transformer = make_column_transformer( ( OneHotEncoder(categories='auto',sparse=False),categorical_features_idx),
    (StandardScaler(), continuous_features_idx),
                        remainder='passthrough'
                        )
    this.transformer.fit( data_table.X )
    # self.transformer = lambda x : self.preprocessor.transform(x).toarray()
    return

def simple_objective(solution,X,Y,lambda_parameter=0.01,target_class_idx=1):
    #
    # for r in solution:
    #     curr_covered_or_not |= r.evaluate_data(X)
    if len(solution) > 0:
        tmp = np.stack([r.evaluate_data(X) for r in solution])
    else:
        tmp = np.zeros(X.shape[0], dtype=np.bool)
    curr_covered_or_not = np.any(tmp,axis=0)
    corret_or_not = np.equal(Y,target_class_idx)

    # theta = sklearn.metrics.accuracy_score(corret_or_not,curr_covered_or_not)
    hit_or_not = np.equal(corret_or_not,curr_covered_or_not)
    theta = np.count_nonzero(hit_or_not) / hit_or_not.shape[0]


    return theta - lambda_parameter * len(solution)

def sampling_criteria(distance_to_nearest_neighnour):
    '''
    simple version is only distance
    '''
    s = distance_to_nearest_neighnour
    return s

def extend_rule(rule,domain):
    from structure import Condition
    tmp_rule = deepcopy(rule)
    specified_columns = [ c.column for c in tmp_rule.conditions  ]
    for attri_col, attribute in enumerate(domain.attributes):
        if attri_col not in specified_columns:
            if attribute.is_discrete:
                new_condition = Condition(column=attri_col,values= [ i for i in range(len(attribute.values)) ],type='categorical')
            else:
                new_condition = Condition(column=attri_col,max_value=attribute.max,min_value=attribute.min,type='continuous',domain=domain)
            tmp_rule.conditions.append(new_condition)
    tmp_rule.conditions = sorted(tmp_rule.conditions,key=lambda x:x.column )
    return tmp_rule

def uniform_sampling(rule_to_uniform_sample,population_size=1000):
    ''' uniform sampling. used as the basic procedure'''
    raw_X_columns = [ s.sample(batch_size=population_size) for s in rule_to_uniform_sample.conditions]
    raw_X = np.column_stack(raw_X_columns)
    return raw_X

def sample_new_instances(a_star,a_prime,X,Y,domain,blackbox,batch_size=1,population_size = 100):

    # TODO: now these two regions are only two rules

    def sample_new_for_one_rule(rule,region,previous_X,previous_Y):
        ''' the true sampling procedure
            1. we first extend the specfic region we wish to sampling
                    in order to sample, first extend the rule
                    for example, if the rule is 1<x<10 -> yes and the domain is 0<x<100,0<y<100
                    then the extended rule is 1<x<10,0<y<100 -> yes
                    in other words, the un-specified conditions are made to be specific
            2. then we sample one new synthetic at a time and we do it for `batch_size` times
        '''
        synthetic_rows = []
        synthetic_rows_y = []

        # tmp_rule = extend_rule(rule,domain)
        tmp_rule = region

        curr_covered_or_not = np.zeros(previous_X.shape[0], dtype=np.bool)
        curr_covered_or_not |= tmp_rule.evaluate_data(previous_X)
        covered_indices = np.where(curr_covered_or_not == True)[0]
        uncovered_indices = np.where(curr_covered_or_not == False)[0]
        population = uniform_sampling(tmp_rule,population_size=population_size)

        if covered_indices.shape[0] == 0 or uncovered_indices.shape[0] == 0:
            '''
            if in this rule there is no instance (which indicates a large upper bound) or it covers all. (Strange and extreme case)
            '''
            synthetic_instances = population[:batch_size]
            synthetic_instances_y = blackbox(synthetic_instances)
            return synthetic_instances,synthetic_instances_y


        specified_columns = [c.column for c in rule.conditions]
        # todo: ugly code change it!
        unspecified_columns = [c.column for c in extend_rule(rule,domain).conditions if c.column not in specified_columns ]
        # mask_data = population[0,unspecified_columns]
        # mask_data_idx = random.choice(uncovered_indices.tolist())
        mask_data_idx = np.random.choice(uncovered_indices, population_size,replace=True )
        # mask_data = previous_X[mask_data_idx,unspecified_columns]
        # population[:,unspecified_columns] = mask_data
        t = previous_X[mask_data_idx]
        population[:,unspecified_columns] =t[:,unspecified_columns]

        population = population.tolist()
        previous_X_copy = [previous_X[i] for i in covered_indices ]
        previous_Y_copy = [previous_Y[i] for i in covered_indices]

        for _ in range(batch_size):

            # transformed_population = np.asarray(population).astype(float)
            # transformed_previous_X_copy = np.asarray(previous_X_copy).astype(float)
            # # print(transformed_population.shape)
            # transformed_population = Table.from_numpy(domain,X=np.asarray(population),Y=np.zeros(len(population), dtype=np.bool))
            # transformed_previous_X_copy = Table.from_numpy(domain,X=np.asarray(previous_X_copy).astype(float),Y=np.zeros(len(previous_X_copy), dtype=np.bool))

            transformed_population = this.transformer.transform( np.asarray(population).astype(float) )
            transformed_previous_X_copy = this.transformer.transform( np.asarray(previous_X_copy).astype(float) )
            tmp = distance_function(transformed_population,transformed_previous_X_copy )
            # tmp = this.distance_model(e1=transformed_population, e2=transformed_previous_X_copy)
            # tmp = this.distance_model.compute_distances(transformed_population,transformed_previous_X_copy)

            nearest_distances = np.amin(tmp, axis=1).tolist()


            idx_to_add = max([ (i,d) for i,d in enumerate(nearest_distances)], key = lambda x: sampling_criteria(x[1]) )[0]
            # TODO: change this

            synthetic_rows.append(population[idx_to_add])
            previous_X_copy.append(population[idx_to_add])
            synthetic_rows_y.append(blackbox([population[idx_to_add]])[0])
            previous_Y_copy.append(blackbox([population[idx_to_add]])[0])

            del population[idx_to_add]


        del previous_X_copy
        synthetic_instances = np.row_stack(synthetic_rows)

        synthetic_instances_y = np.row_stack(synthetic_rows_y).reshape((-1))

        return synthetic_instances,synthetic_instances_y

    region_1,region_2 = get_symmetric_difference(a_star,a_prime,domain)
    if region_1 is None or region_2 is None:
        print("strange! there is a none region")
    new_X_1,new_Y_1 = sample_new_for_one_rule(a_star.changed_rule,region_1,X,Y)
    X_new = np.concatenate((X,new_X_1) )
    Y_new = np.concatenate((Y,new_Y_1) )
    new_X_2,new_Y_2 = sample_new_for_one_rule(a_prime.changed_rule,region_2,X_new,Y_new)
    result = np.concatenate((new_X_1, new_X_2)),np.concatenate((new_Y_1, new_Y_2))

    return result

def get_symmetric_difference(a_1,a_2,domain):
    '''
    given two instances, generate two regions of the symmetric difference
    the returned two regions takes the data structure of a rule. (but is in fact not a rule)
    Not that it is possible that one region is empty.
    '''
    # TODO:change here
    # if a_1.mode == "REMOVE_RULE":
    #     region_1 = a_1.current_solution[a_1.remove_rule_idx]
    # else :
    #     region_1 = a_1.new_solution[-1]
    # if a_2.mode == "REMOVE_RULE":
    #     region_2 = a_2.current_solution[a_2.remove_rule_idx]
    # else:
    #     region_2 = a_2.new_solution[-1]
    tmp_rule_1 = extend_rule(a_1.changed_rule,domain)
    tmp_rule_2 = extend_rule(a_2.changed_rule,domain)

    # region_1 = tmp_rule_1 - tmp_rule_2
    # region_2 = tmp_rule_2 - tmp_rule_1
    region_1 = tmp_rule_1
    region_2 = tmp_rule_2
    # from utils import rule_to_string
    # print(rule_to_string(a_1.changed_rule,domain=domain,target_class_idx=1))
    # print(rule_to_string(a_2.changed_rule,domain=domain,target_class_idx=1))
    # print(rule_to_string(tmp_rule_1,domain=domain,target_class_idx=1))
    # print(rule_to_string(tmp_rule_2,domain=domain,target_class_idx=1))

    # tmp_rule_1 = extend_rule(a_1.changed_rule,domain)
    # tmp_rule_2 = extend_rule(a_2.changed_rule,domain)
    #
    # region_1 = tmp_rule_1 - tmp_rule_2
    # region_2 = tmp_rule_2 - tmp_rule_1
    # if region_1 is not None:
    #     print("r 1 not none")
    #     print(rule_to_string(region_1,domain=domain,target_class_idx=1))
    # else:
    #     print("is none")
    # if region_2 is not None:
    #     print("r 2 not none")
    #     print(rule_to_string(region_2,domain=domain,target_class_idx=1))
    # else:
    #     print("is none")

    # print('***')
    return region_1,region_2

def get_recall(solution,X,Y,target_class_idx=1):

    # curr_covered_or_not = np.zeros(X.shape[0], dtype=np.bool)
    # for r in solution:
    #     curr_covered_or_not |= r.evaluate_data(X)
    curr_covered_or_not = np.logical_or.reduce( [r.evaluate_data(X) for r in solution] )
    corret_or_not = np.equal(Y,target_class_idx)
    return sklearn.metrics.recall_score(corret_or_not,curr_covered_or_not)

def get_incorrect_cover_ruleset(solution,X,Y,target_class_idx=1):
    # curr_covered_or_not = np.zeros(X.shape[0], dtype=np.bool)
    # for r in solution:
    #     curr_covered_or_not |= r.evaluate_data(X)
    curr_covered_or_not = np.logical_or.reduce( [r.evaluate_data(X) for r in solution] )
    corret_or_not = np.equal(Y,target_class_idx)
    return np.where(corret_or_not != curr_covered_or_not)[0]

def get_correct_cover_ruleset(solution,X,Y,target_class_idx=1):
    # curr_covered_or_not = np.zeros(X.shape[0], dtype=np.bool)
    # for r in solution:
    #     curr_covered_or_not |= r.evaluate_data(X)
    curr_covered_or_not = np.logical_or.reduce( [r.evaluate_data(X) for r in solution] )
    corret_or_not = np.equal(Y,target_class_idx)
    return np.where(corret_or_not == curr_covered_or_not)[0]

def best_and_second_best_action(actions):
    # try:
    #     tmp = heapq.nlargest(2,actions)
    #     return tmp[0],tmp[1]
    # except:
    #     # print("strange thing happened, there is only one action")
    #     #
    #     # print("num of actions",len(actions))
    #     null_action = deepcopy(tmp[0]); null_action.make_hopeless()
    #     return tmp[0],null_action
    a_star_idx,a_star = max(enumerate(actions), key=lambda x:x[1].obj_estimation())
    # rest_actions = deepcopy(actions).remove(a_star)
    # rest_actions = actions
    # rest_actions = [a for i,a in enumerate(actions) if i!=a_star_idx]
    rest_actions = [a for a in actions if a!=a_star]
    if len(rest_actions) == 0 :
        a_prime = None
    else:
        a_prime = max(rest_actions,key=lambda x:x.upper())
    return a_star,a_prime
