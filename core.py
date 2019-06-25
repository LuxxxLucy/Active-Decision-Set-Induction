# import general package function
import numpy as np
import heapq
import math
import random
from copy import deepcopy
# import local function
# from objective import objective

from sklearn.neighbors import KDTree

def simple_objective(solution,X,Y,parameter=0.05,target_class_idx=1):
    curr_covered_or_not = np.zeros(X.shape[0], dtype=np.bool)

    for r in solution:
        curr_covered_or_not |= r.evaluate_data(X)

    corret_or_not = np.equal(Y,target_class_idx)
    positive_indices = np.where(Y == target_class_idx)[0]
    covered_indices = np.where(curr_covered_or_not == True)[0]
    not_covered_indices = np.where(curr_covered_or_not != True)[0]

    if covered_indices.shape[0] != 0:
        term_1 = np.intersect1d(positive_indices,covered_indices ).shape[0] / covered_indices.shape[0]
    else:
        term_1 = 0
    if not_covered_indices.shape[0] != 0:
        term_2 = np.intersect1d(positive_indices,not_covered_indices ).shape[0] / not_covered_indices.shape[0]
    else:
        term_2 = 0

    theta =  term_1 - term_2
    return theta - parameter * len(solution)

def sampling_criteria(x,neighbours,neighbors_distances,X,Y):
    '''
    simple version is only distance
    '''

    y_estimated = sum([  (1/dist) * Y[idx] for idx,dist in zip(neighbours,neighbors_distances) ]) / sum([1/dist for dist in neighbors_distances])
    # s = neighbors_distances[0] + (y_estimated - 0.5 )
    s = 10*neighbors_distances[0] - abs( y_estimated - 0.5 )
    return s

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

def extend_rule(rule,domain):
    from structure import Condition
    tmp_rule = deepcopy(rule)
    specified_columns = [ c.column for c in tmp_rule.conditions  ]
    for attri_col, attribute in enumerate(domain.attributes):
        if attri_col not in specified_columns:
            if attribute.is_discrete:
                new_condition = Condition(column=attri_col,values= [ i for i in range(len(attribute.values)) ],type='categorical')
            else:
                new_condition = Condition(column=attri_col,max_value=attribute.max,min_value=attribute.min,type='continuous')
            tmp_rule.conditions.append(new_condition)
    tmp_rule.conditions = sorted(tmp_rule.conditions,key=lambda x:x.column )
    return tmp_rule

def uniform_sampling(rule_to_uniform_sample,population_size=1000,):
    ''' uniform sampling. used as the basic procedure'''
    raw_X_columns = [ s.sample(batch_size=population_size) for s in rule_to_uniform_sample.conditions]
    raw_X = np.column_stack(raw_X_columns).tolist()
    return raw_X

def sample_new_instances(region_1,region_2,X,Y,domain,blackbox,batch_size=10,population_size = 1000):

    # TODO: now these two regions are only two rules

    def sample_new_for_one_rule(rule,previous_X,previous_Y):
        ''' the true sampling procedure
            1. we first extend the specfic region we wish to sampling
                    in order to sample, first extend the rule
                    for example, if the rule is 1<x<10 -> yes and the domain is 0<x<100,0<y<100
                    then the extended rule is 1<x<10,0<y<100 -> yes
                    in other words, the un-specified conditions are made to be specific
            2. then we sample one new synthetic at a time and we do it for `batch_size` times
        '''
        tmp_rule = extend_rule(rule,domain)

        synthetic_rows = []
        synthetic_rows_y = []

        # # todo remove visualization
        # import matplotlib.pyplot as plt
        # import numpy as np
        # previous_X_copy = [X[i] for i in covered_indices ]
        # previous_Y_copy = [Y[i] for i in covered_indices]
        # population = uniform_sampling(tmp_rule)
        # previous_X_copy = np.asarray(previous_X_copy)
        # population_np = np.asarray(population)
        # # previous_Y_copy = np.asarray(previous_Y_copy)
        # fig, ax = plt.subplots()
        # plt.ylim((0,2))
        # plt.xlim((0,1))
        #
        # ax.scatter(previous_X_copy[:,0], previous_X_copy[:,1],c=previous_Y_copy, alpha=0.5)
        # # ax.scatter(population_np[:,0], population_np[:,1], alpha=0.5)
        # # visualize the rule
        # xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 2, 100))
        # output_rule = tmp_rule
        # # Z = predict_by_rule(df_instances, output_rule)
        # Z =  output_rule.evaluate_data(np.c_[xx.ravel(), yy.ravel()])
        #
        # Z = Z.reshape(xx.shape).astype(int)
        # CS2 = ax.contour(xx, yy, Z, cmap=plt.cm.Blues)
        # plt.show()

        previous_X_copy = [x for x in previous_X ]
        previous_Y_copy = [y for y in previous_Y ]
        population = uniform_sampling(tmp_rule,population_size=population_size)

        for _ in range(batch_size):
            # compute the best idx
            # TODO: using more quick K-nearest neighbour

            # a list of pairs in the form of (index, distance to its nearest neighbour)
            if len(previous_X_copy) <=1:
                idx_to_add = random.choice( range(population_size) )
            else:
                from sklearn.neighbors import KDTree

                knn = KDTree(np.asarray(previous_X_copy))
                K = min(5,len(previous_X_copy))
                distances,indices = knn.query(population, k=K)

                # idx_distance_pairs =[ (idx, min([ distance(x, previous_x,domain) for previous_x in previous_X_copy  ]) ) for idx,x in enumerate(population)  ]
                #
                # idx_to_add,_ = max(idx_distance_pairs,key = lambda x:x[1])
                # idx_to_add = max([ (i,d) for i,d in enumerate(distances)], key = lambda x:x[1][0])[0]
                idx_to_add = max([ (i,d) for i,d in enumerate(distances)], key = lambda x: sampling_criteria(x[0],indices[x[0]], distances[x[0]],previous_X_copy,previous_Y_copy) )[0]
                # TODO: change this

            synthetic_rows.append(population[idx_to_add])
            previous_X_copy.append(population[idx_to_add])
            synthetic_rows_y.append(blackbox([population[idx_to_add]])[0])
            previous_Y_copy.append(blackbox([population[idx_to_add]])[0])
            del population[idx_to_add]

        # previous_X_copy_np = np.asarray(previous_X_copy)
        # synthetic_rows_np = np.asarray(synthetic_rows)
        # previous_Y_copy = np.asarray(previous_Y_copy)
        # import matplotlib.pyplot as plt
        # import numpy as np
        # fig, ax = plt.subplots()
        # plt.ylim((0,2))
        # plt.xlim((0,1))
        #
        # ax.scatter(previous_X_copy_np[:,0], previous_X_copy_np[:,1], c=previous_Y_copy , alpha=0.1)
        # ax.scatter(synthetic_rows_np[:,0], synthetic_rows_np[:,1],c=synthetic_rows_y,marker='*',alpha=1)
        # # visualize the rule
        # xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 2, 100))
        # output_rule = tmp_rule
        # # Z = predict_by_rule(df_instances, output_rule)
        # Z =  output_rule.evaluate_data(np.c_[xx.ravel(), yy.ravel()])
        #
        # Z = Z.reshape(xx.shape).astype(int)
        # CS2 = ax.contour(xx, yy, Z, cmap=plt.cm.Blues)
        # plt.show()
        # quit()

        del previous_X_copy
        synthetic_instances = np.row_stack(synthetic_rows)

        synthetic_instances_y = np.row_stack(synthetic_rows_y).reshape((-1))

        return synthetic_instances,synthetic_instances_y

    new_X_1,new_Y_1 = sample_new_for_one_rule(region_1,X,Y)
    X_new = np.concatenate((new_X_1,X) )
    Y_new = np.concatenate((new_Y_1,Y) )
    new_X_2,new_Y_2 = sample_new_for_one_rule(region_2,X_new,Y_new)
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
    if a_1.mode == "REMOVE_RULE":
        region_1 = a_1.current_solution[a_1.remove_rule_idx]
    else :
        region_1 = a_1.new_solution[-1]
    if a_2.mode == "REMOVE_RULE":
        region_2 = a_2.current_solution[a_2.remove_rule_idx]
    else:
        region_2 = a_2.new_solution[-1]
    return region_1,region_2

def sample_new_instances_for_solution(sol_1,sol_2,X,Y,domain,blackbox,batch_size=10,population_size = 1000):

    # TODO: now these two regions are only two rules

    def sample_new_for_one_solution(solution,previous_X,previous_Y):
        ''' the true sampling procedure
            1. we first extend the specfic region we wish to sampling
                    in order to sample, first extend the rule
                    for example, if the rule is 1<x<10 -> yes and the domain is 0<x<100,0<y<100
                    then the extended rule is 1<x<10,0<y<100 -> yes
                    in other words, the un-specified conditions are made to be specific
            2. then we sample one new synthetic at a time and we do it for `batch_size` times
        '''
        tmp_rules = [ extend_rule(rule,domain) for rule in solution.current_solution ]

        synthetic_rows = []
        synthetic_rows_y = []
        # covered_indices = set( [tmp_rule.get_cover(previous_X).tolist() for tmp_rule in tmp_rules] )
        # curr_covered_or_not = np.zeros(X.shape[0], dtype=np.bool)
        # for r in tmp_rules:
        #     curr_covered_or_not |= r.evaluate_data(X)
        # covered_indices = np.where(curr_covered_or_not == True)[0]

        previous_X_copy = [x for x in previous_X ]
        previous_Y_copy = [y for y in previous_Y ]

        popu_list = []
        for tmp_rule in tmp_rules:
            population_tmp = uniform_sampling(tmp_rule)
            popu_list.append(population_tmp)
        population = np.vstack(popu_list).tolist()
        # population = uniform_sampling(tmp_rule,population_size=population_size)

        # # todo remove visualization
        # import matplotlib.pyplot as plt
        # import numpy as np
        # previous_X_copy = [X[i] for i in covered_indices ]
        # previous_Y_copy = [Y[i] for i in covered_indices]
        # population = uniform_sampling(tmp_rule)
        # previous_X_copy = np.asarray(previous_X_copy)
        # population_np = np.asarray(population)
        # # previous_Y_copy = np.asarray(previous_Y_copy)
        # fig, ax = plt.subplots()
        # plt.ylim((0,2))
        # plt.xlim((0,1))
        #
        # ax.scatter(previous_X_copy[:,0], previous_X_copy[:,1],c=previous_Y_copy, alpha=0.5)
        # # ax.scatter(population_np[:,0], population_np[:,1], alpha=0.5)
        # # visualize the rule
        # xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 2, 100))
        # output_rule = tmp_rule
        # # Z = predict_by_rule(df_instances, output_rule)
        # Z =  output_rule.evaluate_data(np.c_[xx.ravel(), yy.ravel()])
        #
        # Z = Z.reshape(xx.shape).astype(int)
        # CS2 = ax.contour(xx, yy, Z, cmap=plt.cm.Blues)
        # plt.show()


        for _ in range(batch_size):
            # compute the best idx
            # TODO: using more quick K-nearest neighbour

            # a list of pairs in the form of (index, distance to its nearest neighbour)
            if len(previous_X_copy) <=1:
                idx_to_add = random.choice( range(population_size) )
            else:
                from sklearn.neighbors import KDTree

                knn = KDTree(np.asarray(previous_X_copy))
                K = min(5,len(previous_X_copy))
                distances,indices = knn.query(population, k=K)

                # idx_distance_pairs =[ (idx, min([ distance(x, previous_x,domain) for previous_x in previous_X_copy  ]) ) for idx,x in enumerate(population)  ]
                #
                # idx_to_add,_ = max(idx_distance_pairs,key = lambda x:x[1])
                # idx_to_add = max([ (i,d) for i,d in enumerate(distances)], key = lambda x:x[1][0])[0]
                idx_to_add = max([ (i,d) for i,d in enumerate(distances)], key = lambda x: sampling_criteria(x[0],indices[x[0]], distances[x[0]],previous_X_copy,previous_Y_copy) )[0]
                # TODO: change this

            synthetic_rows.append(population[idx_to_add])
            previous_X_copy.append(population[idx_to_add])
            synthetic_rows_y.append(blackbox([population[idx_to_add]])[0])
            previous_Y_copy.append(blackbox([population[idx_to_add]])[0])
            del population[idx_to_add]

        # previous_X_copy_np = np.asarray(previous_X_copy)
        # synthetic_rows_np = np.asarray(synthetic_rows)
        # previous_Y_copy = np.asarray(previous_Y_copy)
        # import matplotlib.pyplot as plt
        # import numpy as np
        # fig, ax = plt.subplots()
        # plt.ylim((0,2))
        # plt.xlim((0,1))
        #
        # ax.scatter(previous_X_copy_np[:,0], previous_X_copy_np[:,1], c=previous_Y_copy , alpha=0.1)
        # ax.scatter(synthetic_rows_np[:,0], synthetic_rows_np[:,1],c=synthetic_rows_y,marker='*',alpha=1)
        # # visualize the rule
        # xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 2, 100))
        # output_rule = tmp_rule
        # # Z = predict_by_rule(df_instances, output_rule)
        # Z =  output_rule.evaluate_data(np.c_[xx.ravel(), yy.ravel()])
        #
        # Z = Z.reshape(xx.shape).astype(int)
        # CS2 = ax.contour(xx, yy, Z, cmap=plt.cm.Blues)
        # plt.show()
        # quit()

        del previous_X_copy
        synthetic_instances = np.row_stack(synthetic_rows)
        synthetic_instances_y = np.row_stack(synthetic_rows_y).reshape((-1))

        return synthetic_instances,synthetic_instances_y

    new_X_1,new_Y_1 = sample_new_for_one_solution(sol_1,X,Y)
    X_new = np.concatenate((new_X_1,X) )
    Y_new = np.concatenate((new_Y_1,Y) )
    new_X_2,new_Y_2 = sample_new_for_one_solution(sol_2,X_new,Y_new)
    return np.concatenate((new_X_1, new_X_2))

def get_incorrect_cover_ruleset(solution,X,Y,target_class_idx=1):
    curr_covered_or_not = np.zeros(X.shape[0], dtype=np.bool)
    for r in solution:
        curr_covered_or_not |= r.evaluate_data(X)
    corret_or_not = np.equal(Y,target_class_idx)
    return np.where(corret_or_not != curr_covered_or_not)[0]

def get_correct_cover_ruleset(solution,X,Y,target_class_idx=1):
    curr_covered_or_not = np.zeros(X.shape[0], dtype=np.bool)
    for r in solution:
        curr_covered_or_not |= r.evaluate_data(X)
    corret_or_not = np.equal(Y,target_class_idx)
    return np.where(corret_or_not == curr_covered_or_not)[0]

def best_and_second_best_action(actions):
    try:
        tmp = heapq.nlargest(2,actions)
        return tmp[0],tmp[1]
    except:
        # print("strange thing happened, there is only one action")
        #
        # print("num of actions",len(actions))
        null_action = deepcopy(tmp[0]); null_action.make_hopeless()
        return tmp[0],null_action

def best_and_second_best_candidate(candidates):
    try:
        tmp = heapq.nlargest(2,candidates)
        return tmp[0],tmp[1]
    except:
        null_candidate = deepcopy(tmp[0]); null_candidate.make_hopeless()
        return tmp[0],null_candidate
