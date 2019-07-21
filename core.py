# import general package function
import numpy as np
import heapq
import math
import random
from copy import deepcopy
import time
# import local function
# from objective import objective

from sklearn.neighbors import KDTree
import sklearn
from sklearn.metrics.pairwise import euclidean_distances as distance_function

from sklearn.gaussian_process import GaussianProcessRegressor

def simple_objective(solution,X,Y,lambda_parameter=0.01,target_class_idx=1):
    curr_covered_or_not = np.zeros(X.shape[0], dtype=np.bool)

    for r in solution:
        curr_covered_or_not |= r.evaluate_data(X)

    corret_or_not = np.equal(Y,target_class_idx)
    theta = sklearn.metrics.accuracy_score(corret_or_not,curr_covered_or_not)
    # positive_indices = np.where(Y == target_class_idx)[0]
    # covered_indices = np.where(curr_covered_or_not == True)[0]
    # not_covered_indices = np.where(curr_covered_or_not != True)[0]
    #
    # if covered_indices.shape[0] != 0:
    #     term_1 = np.intersect1d(positive_indices,covered_indices ).shape[0] / covered_indices.shape[0]
    # else:
    #     term_1 = 0
    # if not_covered_indices.shape[0] != 0:
    #     term_2 = np.intersect1d(positive_indices,not_covered_indices ).shape[0] / not_covered_indices.shape[0]
    # else:
    #     term_2 = 0
    #
    # theta =  term_1 - term_2
    return theta - lambda_parameter * len(solution)

def sampling_criteria(distance_to_nearest_neighnour):
    '''
    simple version is only distance
    '''
    # y_estimated = sum([  (1/dist) * 2 * ( Y[idx] -0.5)  for idx,dist in zip(neighbours,neighbors_distances) ]) / sum([1/dist for dist in neighbors_distances])

    s = distance_to_nearest_neighnour
    # try:
    #     y_variance = 1 / (1+math.exp(distance_ratio))
    # except:
    #     print(neighbors_distances)
    #     print(average_nearest_distance)
    #     print(distance_ratio)
    #     quit()


    # X = [X[idx] for idx in neighbours]
    # X = transformer(np.asarray(X))
    # y = [Y[idx] for idx in neighbours]
    # # start_time = time.time()
    # gpr = GaussianProcessRegressor(random_state=42).fit(X,y)
    # # print ('\tTook %0.3fs to generate the local gpr' % (time.time() - start_time ) )
    #
    # predition = gpr.predict( transformer(np.asarray([x])),return_std=True )
    # mean,variance = predition
    # y_estimated = mean[0]
    # y_variance = variance[0]

    # s = 0.000001*y_variance - abs( y_estimated - 0.5 )
    # s =  - abs( y_estimated - 0.5 )
    # s = 1.96*y_variance  - abs( y_estimated )


    # s = distance_to_nearest_neighnour
    # s = - abs( y_estimated - 0.5 )
    # s = np.random.random()
    # s = 10*distance_to_nearest_neighnour - abs( y_estimated - 0.5 )
    # s = 1.96*distance_to_nearest_neighnour - abs( y_estimated - 0.5 )
    # s = 0.01*distance_to_nearest_neighnour - abs( y_estimated - 0.5 )
    # s = 10*neighbors_distances[0]
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
                new_condition = Condition(column=attri_col,max_value=attribute.max,min_value=attribute.min,type='continuous')
            tmp_rule.conditions.append(new_condition)
    tmp_rule.conditions = sorted(tmp_rule.conditions,key=lambda x:x.column )
    return tmp_rule

def uniform_sampling(rule_to_uniform_sample,population_size=1000):
    ''' uniform sampling. used as the basic procedure'''
    raw_X_columns = [ s.sample(batch_size=population_size) for s in rule_to_uniform_sample.conditions]
    raw_X = np.column_stack(raw_X_columns).tolist()
    return raw_X

def sample_new_instances(region_1,region_2,X,Y,domain,blackbox,batch_size=1,population_size = 1000,transformer=None):

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
        # tmp_rule = rule

        synthetic_rows = []
        synthetic_rows_y = []

        population = uniform_sampling(tmp_rule,population_size=population_size)

        curr_covered_or_not = np.zeros(previous_X.shape[0], dtype=np.bool)
        curr_covered_or_not |= tmp_rule.evaluate_data(previous_X)
        covered_indices = np.where(curr_covered_or_not == True)[0]

        if covered_indices.shape[0] == 0:
            '''
            if in this rule there is no instance (which indicates a large upper bound)
            '''
            # print("special case!!!")
            # from utils import rule_to_string
            # print(rule_to_string(rule,domain=domain,target_class_idx=1))
            # print(rule_to_string(tmp_rule,domain=domain,target_class_idx=1))
            synthetic_instances = population[:batch_size]
            synthetic_instances_y = blackbox(synthetic_instances)
            return synthetic_instances,synthetic_instances_y

        # todo remove visualization
        # import matplotlib.pyplot as plt
        # import numpy as np
        # previous_X_copy = [previous_X[i] for i in covered_indices ]
        # previous_Y_copy = [previous_Y[i] for i in covered_indices]
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

        previous_X_copy = [previous_X[i] for i in covered_indices ]
        previous_Y_copy = [previous_Y[i] for i in covered_indices]

        for _ in range(batch_size):
            # compute the best idx
            # TODO: using more quick K-nearest neighbour
            # a list of pairs in the form of (index, distance to its nearest neighbour)
            # knn = KDTree(,metric='euclidean')
            # K = min(30,covered_indices.shape[0])
            # distances,indices = knn.query(, k=K)
            # distances = [ d.tolist() for d in distances]
            # indices = [idx.tolist() for idx in indices ]

            tmp = distance_function(transformer(np.asarray(population)), transformer(np.asarray(previous_X_copy)))
            nearest_distances = np.amin(tmp, axis=1).tolist()

            idx_to_add = max([ (i,d) for i,d in enumerate(nearest_distances)], key = lambda x: sampling_criteria(x[1]) )[0]
            # TODO: change this

            synthetic_rows.append(population[idx_to_add])
            previous_X_copy.append(population[idx_to_add])
            synthetic_rows_y.append(blackbox([population[idx_to_add]])[0])
            previous_Y_copy.append(blackbox([population[idx_to_add]])[0])
            del population[idx_to_add]

        # previous_X_copy_np = np.asarray(previous_X_copy)
        # synthetic_rows_np = np.asarray(synthetic_rows)
        # print("#new instances",synthetic_rows_np.shape)
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

    if region_1 is None or region_2 is None:
        print("strange! there is a none region")
    new_X_1,new_Y_1 = sample_new_for_one_rule(region_1,X,Y)
    X_new = np.concatenate((new_X_1,X) )
    Y_new = np.concatenate((new_Y_1,Y) )
    new_X_2,new_Y_2 = sample_new_for_one_rule(region_2,X_new,Y_new)
    return np.concatenate((new_X_1, new_X_2)),np.concatenate((new_Y_1, new_Y_2))


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
    region_1 = a_1.changed_rule
    region_2 = a_2.changed_rule
    from utils import rule_to_string
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

    curr_covered_or_not = np.zeros(X.shape[0], dtype=np.bool)
    for r in solution:
        curr_covered_or_not |= r.evaluate_data(X)
    corret_or_not = np.equal(Y,target_class_idx)
    return sklearn.metrics.recall_score(corret_or_not,curr_covered_or_not)

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
