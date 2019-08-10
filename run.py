from Orange.data import Table
import numpy as np
import sklearn

from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

from prepare_dataset import prepare_adult_dataset,prepare_compas_dataset,prepare_pima_dataset
from prepare_blackbox import train_classifier

from utils import encoder_from_datatable,ruleset_predict,rule_to_string,label_with_blackbox

from approach import explain_tabular


from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score


if __name__ == '__main__':
    method = "bayesian"
    method = "ours"
    random_seed=42
    np.random.seed(random_seed)
    data_table,test_data_table = prepare_compas_dataset()
    # data_table,test_data_table = prepare_pima_dataset()
    # data_table,test_data_table = prepare_adult_dataset()

    black_box_model = train_classifier(data_table,classifier_method='rf',random_seed=random_seed)
    # black_box_model = train_classifier(data_table,classifier_method='dnn',random_seed=random_seed)

    black_box = lambda x: black_box_model.predict(x)

    print('Train acc', accuracy_score(data_table.Y, black_box(data_table.X)))
    print('Test acc', accuracy_score(test_data_table.Y, black_box(test_data_table.X)))
    print('acc on training set', sklearn.metrics.accuracy_score(data_table.Y, black_box(data_table.X)))
    print('f1 on training set', sklearn.metrics.f1_score(data_table.Y, black_box(data_table.X)))
    print('recall on training set', sklearn.metrics.recall_score(data_table.Y, black_box(data_table.X)))
    print('precision on training set', sklearn.metrics.precision_score(data_table.Y, black_box(data_table.X)))

    print('acc on test set', sklearn.metrics.accuracy_score(test_data_table.Y, black_box(test_data_table.X)))
    print('f1 on test set', sklearn.metrics.f1_score(test_data_table.Y, black_box(test_data_table.X)))
    print('recall on test set', sklearn.metrics.recall_score(test_data_table.Y, black_box(test_data_table.X)))
    print('precision on test set', sklearn.metrics.precision_score(test_data_table.Y, black_box(test_data_table.X)))

    predicted_data_table = label_with_blackbox(data_table,black_box)
    predicted_test_data_table = label_with_blackbox(test_data_table,black_box)

    if method == "ours":
        beta = 0.0001
        beta = 0
        lambda_parameter = 0.0001
        explanations,ADS = explain_tabular(predicted_data_table,  black_box, target_class_idx=1, random_seed=42,beta=beta,lambda_parameter= lambda_parameter)

        explanations = ADS.output_the_best(lambda_parameter)

        for e in explanations:
            print(rule_to_string(e,predicted_data_table.domain,1))
        print("number of rules",len(explanations))
        our_prediction = ruleset_predict(explanations,test_data_table.X)
        print('Blackbox and our, f1 score', sklearn.metrics.f1_score(predicted_test_data_table.Y, our_prediction))
        print('Blackbox and our, accuracy', sklearn.metrics.accuracy_score(predicted_test_data_table.Y, our_prediction))
        print('Blackbox and our,recall', sklearn.metrics.recall_score(predicted_test_data_table.Y, our_prediction))
        print('Blackbox and our,precision', sklearn.metrics.precision_score(predicted_test_data_table.Y, our_prediction))

        lambda_candidates = [  0,0.0000001,0.000001,0.00001,0.0001,0.0005,0.001,0.0013,0.0015,0.0018,0.002,0.0025,0.003,0.005,0.006,0.007,0.008,0.01,0.015,0.02,0.03,0.05,0.1,0.15,0.2,0.5 ]
        for lambda_parameter in  lambda_candidates:
            explanations = ADS.output_the_best(lambda_parameter)
            our_prediction = ruleset_predict(explanations,test_data_table.X)
            f1_score = sklearn.metrics.f1_score(predicted_test_data_table.Y, our_prediction)
            acc_score=sklearn.metrics.accuracy_score(predicted_test_data_table.Y, our_prediction)
            rec_score = sklearn.metrics.recall_score(predicted_test_data_table.Y, our_prediction)
            pre_score = sklearn.metrics.precision_score(predicted_test_data_table.Y, our_prediction)
            num = len(explanations)
            num_of_instance = ADS.synthetic_data_table.X.shape[0]

            print("parameter: ",beta,lambda_parameter,"metrics",num,f1_score,acc_score,rec_score,pre_score,num_of_instance)
    elif method == "bayesian":

        import Orange

        disc = Orange.preprocess.Discretize()
        disc.method = Orange.preprocess.discretize.EntropyMDL()
        # disc.method = Orange.preprocess.discretize.EqualFreq(n=3)
        disc_predicted_data_table = disc(predicted_data_table)
        # disc_predicted_test_data_table = disc(predicted_test_data_table)
        disc_predicted_test_data_table = Orange.data.Table.from_table(disc_predicted_data_table.domain, predicted_test_data_table)


        from utils import uniform_enlarge_dataset,estimated_enlarge_dataset

        rate = 1.0 * 49804 / predicted_data_table.X.shape[0]
        # rate = 0
        print("sampling rate",rate)
        new_predicted_data_table = estimated_enlarge_dataset(predicted_data_table,black_box,sampling_rate=rate,random_seed=random_seed)
        print(new_predicted_data_table.X.shape)
        disc_new_predicted_data_table =  Orange.data.Table.from_table(disc_predicted_data_table.domain, new_predicted_data_table)

        from approach import explain_tabular
        explanations,explainer = explain_tabular(disc_new_predicted_data_table, black_box, target_class_idx=1, random_seed=random_seed,beta = 0,use_pre_mined=True, objective = 'bayesian')

        print(len(explanations))
        from utils import  rule_to_string,ruleset_predict
        our_prediction = ruleset_predict(explanations,disc_predicted_test_data_table.X)

        import sklearn
        print('Blackbox and our, acc', sklearn.metrics.accuracy_score(predicted_test_data_table.Y, our_prediction))
        print('Blackbox and our, f1 score', sklearn.metrics.f1_score(predicted_test_data_table.Y, our_prediction))
        print('Blackbox and our,recall', sklearn.metrics.recall_score(predicted_test_data_table.Y, our_prediction))
        print('Blackbox and our,precision', sklearn.metrics.precision_score(predicted_test_data_table.Y, our_prediction))

        from utils import compute_metrics
        compute_metrics(explanations,disc_predicted_data_table.domain)
