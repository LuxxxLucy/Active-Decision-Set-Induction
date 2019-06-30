from Orange.data import Table

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

from sklearn.ensemble import  RandomForestClassifier

from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

from prepare_dataset import prepare_adult_dataset
from utils import encoder_from_datatable,ruleset_predict


from approach import explain_tabular

if __name__ == "__main__":
    random_seed = 42

    data_table,df = prepare_adult_dataset("adult.csv","./datasets/adult/")

    print("now start sanity-check")
    print("the first row of the dataset is (orginal form):\n", data_table[0] )
    print("the first row of the dataset is (only X data):\n", data_table.X[0])
    print("the first row of the dataset is (only y data):\n", data_table.Y[0])

    X_train,X_test,y_train,y_test = train_test_split(data_table.X,data_table.Y,test_size=0.1,random_state=random_seed)

    data_table = Table.from_numpy(X=X_train,Y=y_train,domain=data_table.domain)
    test_data_table = Table.from_numpy(X=X_test,Y=y_test,domain=data_table.domain)
    categorical_features_idx = [i for i,a in enumerate(data_table.domain.attributes) if a.is_discrete]

    scikit_encoder = make_column_transformer(
                                ( OneHotEncoder(categories='auto'),categorical_features_idx),
                                remainder = 'passthrough'
                                )
    scikit_encoder.fit(data_table.X)

    c =RandomForestClassifier(n_estimators=15, n_jobs=5)
    # c = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
    c.fit(scikit_encoder.transform(X_train), y_train)
    black_box = lambda x: c.predict(scikit_encoder.transform(x))

    explanations,ADS = explain_tabular(data_table,  black_box, target_class=">50K", random_seed=42,beta=100)

    for e in explanations:
        print(rule_to_string(e,data_table.domain,1))

    prediction = ruleset_predict(explanations,X_test)

    print('Blackbox and non active ours, acc', sklearn.metrics.accuracy_score(y_test, prediction) )
    print('Blackbox and non active ours, f1', sklearn.metrics.f1_score(y_test, prediction) )

    print('Blackbox and non active ours recall', sklearn.metrics.recall_score(y_test, prediction) )
    print('Blackbox and non active ours precision', sklearn.metrics.precision_score(y_test, prediction) )
