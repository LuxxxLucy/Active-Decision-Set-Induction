from Orange.data import Table
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

from sklearn.ensemble import  RandomForestClassifier

from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

from prepare_dataset import prepare_adult_dataset,prepare_compas_dataset,prepare_pima_dataset
from utils import encoder_from_datatable,ruleset_predict,rule_to_string,label_with_blackbox

from approach import explain_tabular


from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score


if __name__ == '__main__':
    random_seed=42
    np.random.seed(random_seed)
    # data_table,df = prepare_adult_dataset("adult.csv","./datasets/adult/")
    # data_table,test_data_table = prepare_compas_dataset()
    # data_table,test_data_table = prepare_pima_dataset()
    data_table,test_data_table = prepare_adult_dataset()

    categorical_features_idx = [i for i,a in enumerate(data_table.domain.attributes) if a.is_discrete]

    scikit_encoder = make_column_transformer(
                                ( OneHotEncoder(categories='auto'),categorical_features_idx),
                                remainder = 'passthrough'
                                )
    scikit_encoder.fit(data_table.X)

    from sklearn.ensemble import  RandomForestClassifier
    c =RandomForestClassifier(n_estimators=15, n_jobs=5,random_state=random_seed)

    # from sklearn.neural_network import MLPClassifier
    # c = MLPClassifier(solver='adam', alpha=1e-5,  hidden_layer_sizes=(100, 100,50,50,10), random_state=random_seed,verbose=False)

    # c = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

    print("training start")
    c.fit(scikit_encoder.transform(data_table.X), data_table.Y)
    black_box = lambda x:  c.predict(scikit_encoder.transform(x))
    print("training okay")

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

    explanations,ADS = explain_tabular(predicted_data_table,  black_box, target_class_idx=1, random_seed=42,beta=0,lambda_parameter=0.01)

    explanations = ADS.output_the_best(0.01)
    for e in explanations:
        print(rule_to_string(e,predicted_data_table.domain,1))

    our_prediction = ruleset_predict(explanations,test_data_table.X)
    print('Blackbox and our, f1 score', sklearn.metrics.f1_score(predicted_test_data_table.Y, our_prediction))
    print('Blackbox and our, accuracy', sklearn.metrics.accuracy_score(predicted_test_data_table.Y, our_prediction))
    print('Blackbox and our,recall', sklearn.metrics.recall_score(predicted_test_data_table.Y, our_prediction))
    print('Blackbox and our,precision', sklearn.metrics.precision_score(predicted_test_data_table.Y, our_prediction))
