# import general package
import numpy as np
import random
from collections import namedtuple
import operator

# import particular package we utilize
import Orange

# import local package
from model import ADS_Learner
from core import best_and_second_best_action


import logging
import time
import os

def explain_tabular(dataset,blackbox, target_class_idx=1, pre_label=True,verbose=True, random_seed=42,beta=1,lambda_parameter=0.01,use_pre_mined = False, objective = 'simple'):
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
    random.seed(random_seed)
    # os.environ['PYTHONHASHSEED'] = random_seed
    # hash(1)
    # print(os.environ['PYTHONHASHSEED'])
    # quit()
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
    target_class = dataset.domain.class_var.values[target_class_idx]
    explainer = ADS_Learner(dataset, blackbox, target_class=target_class,use_pre_mined=use_pre_mined,objective=objective) # fit the explainer to the data
    # todo: remove this
    explainer.tmp_time = 0
    logging.info('object construct okay')

    print("new cached!")


    logging.info('set hyperparameter: beta and lambda')
    explainer.set_parameters(beta=beta,lambda_parameter=lambda_parameter) # set hyperparameter


    logging.info('initialize the empty decision set')
    explainer.initialize() # initialize the decision set as empty

    logging.info('initialize the synthetic instances')
    explainer.initialize_synthetic_dataset() # initialize synthetic dataset as empty
    # the local search: main algorithm
    logging.info('Begin the outer loop')
    if verbose:
        from tqdm import tqdm_notebook as tqdm
        # from tqdm import tqdm
        pbar = tqdm(total=explainer.N_iter_max)

    # print("not resetting XY")
    for iter_number in range(explainer.N_iter_max):
    # for iter_number in range(explainer.N_iter_max):
        # explainer.reset_XY()
        if verbose:
            pbar.set_description("n_rules "+str(len(explainer.current_solution)))
            pbar.update(1)
        try:
            actions = explainer.generate_action_space() # generating possible actions
        except Exception as e:
            error_log_list.append((iter_number,e))
            continue
        # actions = explainer.generate_action_space() # generating possible actions

        logging.debug("action genearting okay")
        logging.debug("finding best and second best")
        a_star,a_prime = best_and_second_best_action(actions)
        logging.debug("finding best and second best okay")

        # start_time = time.time()
        tmp_count = 0
        if a_prime is None:
            explainer.update_best_solution(a_star) # update best solution
            explainer.update_current_solution(a_star,actions) #take the greedy or take random
            explainer.update() # count += 1
            del actions
            continue

        # while(a_prime.upper() > a_star.lower() ) :
        while(a_prime.upper() > a_star.lower()+0.0001 ) :
            if a_prime == a_star:
                print("dupppp")
                break

            X_new,Y_new = explainer.generate_synthetic_instances(a_star,a_prime)
            solutions =  explainer.update_actions(actions,a_star,a_prime,X_new,Y_new)
            a_star,a_prime = best_and_second_best_action(actions)
            tmp_count+=1
            if tmp_count >= 100:
                break
            if a_prime is None:
                break
        if tmp_count != 0:
            logging.info("generating new instance for a batch : "+str(tmp_count))
            # print("generating new instance for a batch : ",str(tmp_count),"action mode",a_star.mode)
            # print(explainer.synthetic_data_table[-2*(tmp_count):])

        explainer.update_best_solution(a_star) # update best solution
        explainer.update_current_solution(a_star,actions) #take the greedy or take random
        explainer.update() # count += 1
        del actions

    if verbose:
        # pbar.set_description("n_rules "+str(len(explainer.current_solution)))
        pbar.clear()
        pbar.close()

    # explainer.reset_XY()
    rule_set = explainer.output()
    # print("the result accuracy",explainer.compute_accuracy(rule_set))
    # print("the number of rules",len(rule_set))
    # print("the number of new instances generated",len(explainer.synthetic_data_table))
    explainer.error_log_list = error_log_list
    print("Now print Error Log")
    if len(error_log_list) == 0:
        print("no error happens")
    else:
        for iter_num,e in error_log_list:
            print("at iteration:",iter_num,"happens the following error")
            print(e)

    return rule_set,explainer
