
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline

def train_classifier(data_table,classifier_method="dnn",random_seed=42):

    categorical_features_idx = [i for i,a in enumerate(data_table.domain.attributes) if a.is_discrete]
    continuous_features_idx = [i for i,a in enumerate(data_table.domain.attributes) if a.is_continuous]

    scikit_encoder = make_column_transformer( ( OneHotEncoder(categories='auto',sparse=False),categorical_features_idx),
    (StandardScaler(), continuous_features_idx),
                        remainder = 'passthrough'
                        )
    # scikit_encoder.fit(data_table.X)

    if classifier_method == "rf":
        from sklearn.ensemble import  RandomForestClassifier
        c =RandomForestClassifier(n_estimators=15, n_jobs=5,random_state=random_seed)
    elif classifier_method == "dnn":
        from sklearn.neural_network import MLPClassifier
        # c = MLPClassifier(solver='adam', alpha=1e-3,  hidden_layer_sizes=(100,100,50,50,10), random_state=random_seed,verbose=True)
        c = MLPClassifier(solver='adam', alpha=1e-3,  hidden_layer_sizes=(100,100,50,50,10), random_state=random_seed,verbose=False)
    elif classifier_method == "xgb":
        from xgboost import XGBClassifier
        c = XGBClassifier(random_state=random_seed)
    else:
        print("this classifier not included yet")


    blackbox = Pipeline([('encoder', scikit_encoder), ('model', c)])

    blackbox.fit(data_table.X, data_table.Y)

    return blackbox
