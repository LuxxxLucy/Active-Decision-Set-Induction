from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

def encoder_from_datatable(data_table):
    # encoder = collections.namedtuple('random_name',
    #                                       ['fit_transform'])(lambda x: x)

    categorical_features = [a.name for a in data_table.domain.attributes if a.is_discrete]
    categorical_features_idx = [i for i,a in enumerate(data_table.domain.attributes) if a.is_discrete]
    numerical_features = [a.name for a in data_table.domain.attributes if a.is_continuous]
    numerical_features_idx = [i for i,a in enumerate(data_table.domain.attributes) if a.is_continuous]
    preprocess_encoder = make_column_transformer(
                            ( make_pipeline(SimpleImputer(), StandardScaler()), numerical_features_idx  ),
                            ( OneHotEncoder(categories='auto'),categorical_features_idx)
                            )
    preprocess_encoder.fit(data_table.X)
    return preprocess_encoder
