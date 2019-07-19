# import general package
import numpy as np
import random
from collections import namedtuple
import operator

# import particular package we utilize
import Orange

# import local package
from model import ADS
from core import best_and_second_best_action

from tqdm import tqdm_notebook as tqdm
import logging
# from tqdm import tqdm

def explain_tabular(dataset,blackbox, target_class='yes', pre_label=True, random_seed=42,beta=1,lambda_parameter=0.01):
    '''
    Input Params:
    1. dataset: a Orange data table
    2. blackbox: a blackbox predict function, such as `c.predict` where c is a scikit-classifier
    3. target_class
    ---
    Output:
    A decision set.
    '''
    np.random.seed(random_seed)
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.WARNING)

    if pre_label == False:
        logging.info("labelling the dataset with the prediction")
        # re-labelled the data using the blackbox, otherwise assuming the labels provided is labeled by the classifier, (instead of the groundtruth label)
        logging.info("starting labelling")
        labels = blackbox(dataset.X)
        logging.info("labelling okay, start construct the dataset")
        dataset = Orange.data.Table(dataset.domain, dataset.X, labels)
        logging.info("pre-label okay")

    error_log_list = []

    # initialize with the true data instances
    logging.info('construct the object')
    explainer = ADS(dataset, blackbox, target_class=target_class) # fit the explainer to the data
    logging.info('object construct okay')



    logging.info('set hyperparameter: beta and lambda')
    explainer.set_parameters(beta=beta,lambda_parameter=lambda_parameter) # set hyperparameter


    logging.info('initialize the empty decision set')
    explainer.initialize() # initialize the decision set as empty

    logging.info('initialize the synthetic instances')
    explainer.initialize_synthetic_dataset() # initialize synthetic dataset as empty
    # the local search: main algorithm
    logging.info('Begin the outer loop')
    for iter_number in tqdm(range(explainer.N_iter_max)):
        try:
            actions = explainer.generate_action_space() # generating possible actions
        except Exception as e:
            # print(e)
            error_log_list.append((iter_number,e))
            continue
        # actions = explainer.generate_action_space() # generating possible actions
        logging.debug("action genearting okay")
        logging.debug("finding best and second best")
        a_star,a_prime = best_and_second_best_action(actions)
        logging.debug("finding best and second best okay")

        tmp_count = 0
        if a_prime is None:
            explainer.update_best_solution(a_star) # update best solution
            explainer.update_current_solution(a_star,actions) #take the greedy or take random
            explainer.update() # count += 1
            del actions
            continue

        # while(a_prime.upper() > a_star.lower()+0.001 ) :
        while(a_prime.upper() > a_star.lower() + 0.00001  ) :
            # print(a_prime.upper(),a_prime.beta_interval,a_star.lower(),a_star.beta_interval)
            # print(tmp_count)
            # logging.debug("inner loop iter count at "+str(tmp_count))
            # print("action mode: ",a_star.mode)
            # from utils import rule_to_string
            # print("*****")
            # print(a_star.mode,a_star.obj_estimation(),a_star.lower(),a_star.beta_interval,a_star.num,a_star.this_rho)
            # print("changed rule is")
            # print(rule_to_string(a_star.changed_rule,domain=dataset.domain, target_class_idx=1))
            # print(a_prime.mode,a_prime.obj_estimation(),a_prime.upper(),a_prime.beta_interval,a_prime.num,a_prime.this_rho)
            # print("changed rule is")
            # print(rule_to_string(a_prime.changed_rule,domain=dataset.domain, target_class_idx=1))
            # print("*****")
            if a_prime == a_star:
                print("strange thing!!! a star a prime is the same")

            X_new,Y_new = explainer.generate_synthetic_instances(a_star,a_prime)
            solutions =  explainer.update_actions(actions,a_star,a_prime,X_new,Y_new)
            a_star,a_prime = best_and_second_best_action(actions)
            if a_prime is None:
                break
            tmp_count+=1
            if tmp_count >= 100:
                # from utils import rule_to_string
                # # logging.debug(a_star.mode,a_star.total_volume,a_prime.total_volume)
                # logging.debug(rule_to_string(a_star.changed_rule,domain=dataset.domain, target_class_idx=1))
                # logging.debug(rule_to_string(a_prime.changed_rule,domain=dataset.domain, target_class_idx=1))
                # from utils import rule_to_string
                # print(a_star.mode,a_star.obj_estimation(),a_star.lower(),a_star.beta_interval,a_star.num,a_star.this_rho)
                # for e in a_star.current_solution:
                #     print(rule_to_string(e,domain=dataset.domain, target_class_idx=1))
                # print("changed rule is")
                # print(rule_to_string(a_star.changed_rule,domain=dataset.domain, target_class_idx=1))
                # print(hash(a_star))
                # print('.....')
                #
                # print(a_prime.mode,a_prime.obj_estimation(),a_prime.upper(),a_prime.beta_interval,a_prime.num,a_prime.this_rho)
                # for e in a_prime.current_solution:
                #     print(rule_to_string(e,domain=dataset.domain, target_class_idx=1))
                # print("changed rule is")
                # print(rule_to_string(a_prime.changed_rule,domain=dataset.domain, target_class_idx=1))
                # print(hash(a_prime))
                # print("-------")
                # from core import get_symmetric_difference
                # r1,r2 = get_symmetric_difference(a_star,a_prime,dataset.domain)
                # print(rule_to_string(r1,domain=dataset.domain, target_class_idx=1))
                # print(rule_to_string(r2,domain=dataset.domain, target_class_idx=1))

                # print("total number of actions",len(actions))
                #
                # print("more than 100 batch new generated")
                # print("*****")
                # print("*****")
                # print("*****")
                # print("*****")
                break
        if tmp_count != 0:
            logging.info("generating new instance for a batch : "+str(tmp_count))
            print("generating new instance for a batch : ",str(tmp_count))
        if tmp_count >= 25:
            # from utils import rule_to_string
            # # logging.debug(a_star.mode,a_star.total_volume,a_prime.total_volume)
            # logging.debug(rule_to_string(a_star.changed_rule,domain=dataset.domain, target_class_idx=1))
            # logging.debug(rule_to_string(a_prime.changed_rule,domain=dataset.domain, target_class_idx=1))
            from utils import rule_to_string
            print(a_star.mode,a_star.obj_estimation(),a_star.lower(),a_star.beta_interval,a_star.num,a_star.this_rho)
            for e in a_star.current_solution:
                print(rule_to_string(e,domain=dataset.domain, target_class_idx=1))
            print("changed rule is")
            print(rule_to_string(a_star.changed_rule,domain=dataset.domain, target_class_idx=1))
            print(hash(a_star))
            print('.....')

            print(a_prime.mode,a_prime.obj_estimation(),a_prime.upper(),a_prime.beta_interval,a_prime.num,a_prime.this_rho)
            for e in a_prime.current_solution:
                print(rule_to_string(e,domain=dataset.domain, target_class_idx=1))
            print("changed rule is")
            print(rule_to_string(a_prime.changed_rule,domain=dataset.domain, target_class_idx=1))
            print(hash(a_prime))
            print("-------")
            # from core import get_symmetric_difference
            # r1,r2 = get_symmetric_difference(a_star,a_prime,dataset.domain)
            # print(rule_to_string(r1,domain=dataset.domain, target_class_idx=1))
            # print(rule_to_string(r2,domain=dataset.domain, target_class_idx=1))



        explainer.update_best_solution(a_star) # update best solution
        explainer.update_current_solution(a_star,actions) #take the greedy or take random
        explainer.update() # count += 1
        del actions

    rule_set = explainer.output()
    # print("the result accuracy",explainer.compute_accuracy(rule_set))
    # print("the number of rules",len(rule_set))
    # print("the number of new instances generated",len(explainer.synthetic_data_table))
    explainer.error_log_list = error_log_list
    # print("Now print Error Log")
    # if len(error_log_list) == 0:
    #     print("no error happens")
    # else:
    #     for iter_num,e in error_log_list:
    #         print("at iteration:",iter_num,"happens the following error")
    #         print(e)

    return rule_set,explainer
