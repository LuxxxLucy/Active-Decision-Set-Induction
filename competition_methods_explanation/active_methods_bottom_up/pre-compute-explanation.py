
import numpy as np

np.random.seed(0)
import sys
import sklearn
import sklearn.ensemble

from competition_methods_explanation.active_methods_bottom_up.anchor import utils
from competition_methods_explanation.active_methods_bottom_up.anchor import anchor_tabular
import pandas as pd

if __name__ == "__main__":

    df = pd.read_csv("datasets/adult/adult.csv")
    # handling space in the column name
    df = df.rename(columns=lambda x: x.strip())
    df.head()
    target_class = "class"

    from utils import load_dataset_dataframe, encoder_from_dataset

    adult_dataset = load_dataset_dataframe(df, target_class="class",feature_to_use=None, categorical_features=None, balance=False,discretize=True)
    new_encoder = encoder_from_dataset(adult_dataset)

    c = sklearn.ensemble.RandomForestClassifier(n_estimators=50, n_jobs=5)
    c.fit(new_encoder.transform(adult_dataset.train), adult_dataset.labels_train)

    predict_fn = lambda x: c.predict(new_encoder.transform(x))
    print('Train', sklearn.metrics.accuracy_score(adult_dataset.labels_train, predict_fn(adult_dataset.train)))
    print('Test', sklearn.metrics.accuracy_score(adult_dataset.labels_test, predict_fn(adult_dataset.test)))

    #
    from competition_methods_explanation.sp_anchor import submodular_pick_anchor
    print("start pre-computing anchor for each instance")
    submodular_pick_anchor(adult_dataset, new_encoder, c.predict, precompute_explanation=False,file_name = "tmp/adult_anchor.pickle",random_seed=42)
    print("pre-computing anchor for each instance all finished")
    print("test SP-Anchor")
    # submodular_pick_anchor(adult_dataset, new_encoder, c.predict, precompute_explanation=True,file_name = "adult_anchor.pickle",random_seed=42)
    # print("pre-computing anchor for each instance all finished")
