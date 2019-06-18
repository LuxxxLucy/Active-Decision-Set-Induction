# import general package function
import numpy as np
import heapq
import math
import random
from copy import deepcopy
# import local function
# from objective import objective

def distance(x1,x2,domain):
        distance = 0
        for attri_idx,attri in enumerate(domain.attributes):
            if attri.is_discrete:
                # TODO: change categorical distance
                distance +=   1 / (len(attri.values)-1) if x1[attri_idx] != x2[attri_idx] else 0
            else:
                # TODO: change continuous distance
                x1_rescaled = (x1[attri_idx] - attri.min ) / (attri.max - attri.min)
                x2_rescaled = (x2[attri_idx] - attri.min ) / (attri.max - attri.min)
                distance += abs(x1_rescaled - x2_rescaled)

        return distance

def sample_new_instances(region_1,region_2,X,Y,domain,batch_size=10,population_size = 100):
    from structure import Condition
    # TODO: now these two regions are only two rules

    def uniform_sampling(rule_to_uniform_sample):
        ''' uniform sampling. used as the basic procedure'''
        raw_X_columns = [ s.sample(batch_size=population_size) for s in rule_to_uniform_sample.conditions]
        raw_X = np.column_stack(raw_X_columns)
        return raw_X

    def sample_new_for_one_rule(rule,previous_X):
        ''' the true sampling procedure
            1. we first extend the specfic region we wish to sampling
                    in order to sample, first extend the rule
                    for example, if the rule is 1<x<10 -> yes and the domain is 0<x<100,0<y<100
                    then the extended rule is 1<x<10,0<y<100 -> yes
                    in other words, the un-specified conditions are made to be specific
            2. then we sample one new synthetic at a time and we do it for `batch_size` times
        '''
        tmp_rule = deepcopy(rule)
        specified_columns = [ c.column for c in tmp_rule.conditions  ]
        for attri_col, attribute in enumerate(domain.attributes):
            if attri_col not in specified_columns:
                if attribute.is_discrete:
                    new_condition = Condition(column=attri_col,values=attribute.values,type='categorical')
                else:
                    new_condition = Condition(column=attri_col,max_value=attribute.max,min_value=attribute.min,type='continuous')
                tmp_rule.conditions.append(new_condition)
        tmp_rule.conditions = sorted(tmp_rule.conditions,key=lambda x:x.column )

        synthetic_rows = []
        covered_indices = tmp_rule.get_cover(previous_X)

        previous_X_copy = [X[i] for i in covered_indices ]
        for _ in range(batch_size):
            population = uniform_sampling(tmp_rule)

            # compute the best idx
            # TODO: using more quick K-nearest neighbour

            # a list of pairs in the form of (index, distance to its nearest neighbour)
            if len(previous_X_copy) <=1:
                idx_to_add = random.choice( range(population_size) )
            else:
                idx_distance_pairs =[ (idx, min([ distance(x, previous_x,domain) for previous_x in previous_X_copy  ]) ) for idx,x in enumerate(population)  ]

                idx_to_add,_ = max(idx_distance_pairs,key = lambda x:x[1])
            synthetic_rows.append(population[idx_to_add])
            previous_X_copy.append(population[idx_to_add])

        del previous_X_copy
        synthetic_instances = np.row_stack(synthetic_rows)

        return synthetic_instances

    new_X_1 = sample_new_for_one_rule(region_1,X)
    new_X_2 = sample_new_for_one_rule(region_2,X)
    return np.concatenate((new_X_1, new_X_2))

def get_symmetric_difference(a_1,a_2,domain):
    '''
    given two instances, generate two regions of the symmetric difference
    the returned two regions takes the data structure of a rule. (but is in fact not a rule)
    Not that it is possible that one region is empty.
    '''
    # TODO:change here


    # decision_set_1 = a_1.new_solution
    # decision_set_2 = a_2.new_solution
    #
    # from utils import rule_to_string
    # for r in decision_set_1:
    #     print(rule_to_string(r,domain,1))
    # for r in decision_set_2:
    #     print(rule_to_string(r,domain,1))
    #
    # def subtract_decision_set(set_1,set_2):
    #     tmp_set_1 = deepcopy(set_1)
    #     tmp_set_2 = deepcopy(set_2)
    #     for r in tmp_set_2:
    #         if r in tmp_set_1:
    #             tmp_set_1.remove(r)
    #     #         tmp_set_2.remove(r)
    #     # if len(tmp_set_1) == 0:
    #     #     return tmp_set_1
    #     # for r_1 in tmp_set_1:
    #     #     for r_2 in tmp_set_2:
    #     #         for c_r_1_index,c_r_1 in enumerate(r_1.conditions):
    #     #             for c_r_2 in r_2.conditions:
    #     #                 c_r_1 = c_r_1 - c_r_2
    #     #                 if c_r_1 is None:
    #     #                     print('null condition returned')
    #     #                     del r_1.conditions[c_r_1_index]
    #     return tmp_set_1
    #
    # region_1 = subtract_decision_set(decision_set_1,decision_set_2)
    # region_2 = subtract_decision_set(decision_set_2,decision_set_1)

    # if region_1 is None:
    #     print("1 is none!")
    # if region_2 is None:
    #     print("2 is none!")
    #
    # print("print result")
    # print(region_1)
    # for r in region_1:
    #     print(rule_to_string(r,domain,1))
    # print(region_1)
    # for r in region_2:
    #     print(rule_to_string(r,domain,1))
    # print("print result over")
    # quit()
    # assert a_1.mode == a_2.mode,"Assertion Error:two actions have different mode!"
    if a_1.mode in [ "ADD_RULE","REPLACE_RULE" ]:
        region_1 = a_1.new_solution[-1]
    elif a_1.mode == "REMOVE_RULE":
        region_1 = a_1.current_solution[a_1.remove_rule_idx]
    if a_2.mode in [ "ADD_RULE","REPLACE_RULE" ]:
        region_2 = a_2.new_solution[-1]
    elif a_2.mode == "REMOVE_RULE":
        region_2 = a_2.current_solution[a_2.remove_rule_idx]
    return region_1,region_2


def simple_objective(solution,X,Y,parameter=0.01):
    theta = len(get_correct_cover_ruleset(solution,X,Y)) * 1.0 / len(X)
    return theta - parameter * len(solution)

def get_incorrect_cover_ruleset(solution,X,Y):
    curr_covered_or_not = np.zeros(X.shape[0], dtype=np.bool)
    for r in solution:
        curr_covered_or_not |= r.evaluate_data(X)
    return np.where(Y.astype(np.bool)!=curr_covered_or_not)[0]

def get_correct_cover_ruleset(solution,X,Y):
    curr_covered_or_not = np.zeros(X.shape[0], dtype=np.bool)
    for r in solution:
        curr_covered_or_not |= r.evaluate_data(X)
    return np.where(Y.astype(np.bool)==curr_covered_or_not)[0]

def best_and_second_best_action(actions):
    try:
        tmp = heapq.nlargest(2,actions)
        return tmp[0],tmp[1]
    except:
        print("strang thing happened, there is only one action")
        print(len(actions))
        null_action = deepcopy(tmp[0]); null_action.make_hopeless()
        return tmp[0],null_action


# Helper function for smooth_local_search routine: Computes the 'estimate' of optimal value using random search
def compute_OPT(list_rules, X, Y, lambda_array):
    opt_set = set()
    for i in range(len(list_rules)):
        r_val = np.random.uniform()
        if r_val <= 0.5:
            opt_set.add(i)
    return objective(opt_set, list_rules, X, Y, lambda_array)

# Helper function for smooth_local_search routine: Computes estimated gain of adding an element to the solution set
def estimate_omega_for_element(soln_set, delta, rule_x_index, list_rules, df, Y, lambda_array, error_threshold):
    #assumes rule_x_index is not in soln_set

    Exp1_func_vals = []

    Exp2_func_vals = []

    while(True):

        # first expectation term (include x)
        for i in range(10):
            temp_soln_set = sample_random_set(soln_set, delta, len(list_rules))
            temp_soln_set.add(rule_x_index)
            Exp1_func_vals.append(objective(temp_soln_set, list_rules, df, Y, lambda_array))

        # second expectation term (exclude x)
        for j in range(10):
            temp_soln_set = sample_random_set(soln_set, delta, len(list_rules))
            if rule_x_index in temp_soln_set:
                temp_soln_set.remove(rule_x_index)
            Exp2_func_vals.append(objective(temp_soln_set, list_rules, df, Y, lambda_array))

        # compute standard error of mean difference
        variance_Exp1 = np.var(Exp1_func_vals, dtype=np.float64)
        variance_Exp2 = np.var(Exp2_func_vals, dtype=np.float64)
        std_err = math.sqrt(variance_Exp1/len(Exp1_func_vals) + variance_Exp2/len(Exp2_func_vals))
        print("Standard Error "+str(std_err))

        if std_err <= error_threshold:
            break

    return np.mean(Exp1_func_vals) - np.mean(Exp2_func_vals)


# Helper function for smooth_local_search routine: Samples a set of elements based on delta
def sample_random_set(soln_set, delta, len_list_rules):
    all_rule_indexes = set(range(len_list_rules))
    return_set = set()

    # sample in-set elements with prob. (delta + 1)/2
    p = (delta + 1.0)/2
    for item in soln_set:
        random_val = np.random.uniform()
        if random_val <= p:
            return_set.add(item)

    # sample out-set elements with prob. (1 - delta)/2
    p_prime = (1.0 - delta)/2
    for item in (all_rule_indexes - soln_set):
        random_val = np.random.uniform()
        if random_val <= p_prime:
            return_set.add(item)

    #print(soln_set)
    #print(all_rule_indexes - soln_set)
    return return_set
