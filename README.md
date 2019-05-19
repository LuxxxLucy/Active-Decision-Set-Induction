# Descriptive Induction from Machine Learning Model

This is a repo for current research project on descriptive induction from machine learning model.
This is a model-agnostic approach that could be used to infer human-readable patterns from any blackbox model (classifier).

Currently the core code is not included but several competition methods are re-implemented or adopted.

# Requirements
I assume you use anaconda.
* `python3`: all the code and scripts should work on python-3.5. And by a large chance, it would run all fine in python 3.6 and above.
* `scipy`, `numpy`, `scikit-learn`: the packages you should have
* `Orange`: (I hate pandas!) The Orange package has a good representation of data, especially the `Domain` object.
* __lime__: this is the dependency for one baseline method (the Anchor algorithm). run `conda install -c conda-forge lime`
* __spacy__: this is used for processing text data.  


# Some notes for the competition methods.
---
1. `sp_anchor.py`:the Anchor implementation from the original author is slightly modified, about the data preprocessing. (None of the core code is modified). A SP-Anchor function is added.
2. `cn2anchor.py`: warper of separate-and-conquer manner CN2-Anchor algorithm. The original anchor implementation is adopted, which employs a top-down (general-to-specific search with KL-LUCB algorithm).  


# Data Structure
---
* the blackbox model should be a scikit-learn classifier.
* encoder: more specifically a sci-kit column transformer to encode raw data into classifier-readable data (for example, converting categorical data into one-hot-code). Please refer to scikit documentation on ColumnTransformer
* for tabular data, the Orange Table is the data structure we'll be working on. It has several very nice properties compared with Pandas dataframe or scikit-learn dictionary-like 'bunch' object. And that is the concept of "domain". Basically, the types of featrues ("categorical", etc) are explicitly defined in the data table structure. It avoids the tedious and strange use of scikit "bunch" object.

# Experiments

(Note: the notebooks are not included in the git repo for now)

* `sanity-check-dataset-loading-and-one-instance-explanation.ipynb`: sanity check
* `2d-experiment.ipynb`: experiments on 2d sinusoidal data and visualization. -->


# Structure of Code

# Hyper-Parameters

There hyper-parameters for this algorithm:
1. in `mcts.py`: there is a `BASE_BATCH_SIZE`: this indicates how many synthetic instances are generated and queried for one time.
2. in `core.py`:
    1. `THRESHOLD`:  the threshold of precision, 0.01 means that a precision rule is of 99 percent accuracy
    2. `K_INTERVALS`: number of the possible split point for continuous features.
    3. `SMALL_THRESHOLD`: it specifies how small a interval will not be considered not to be splited for continuous variable.
    4. `SMALL_SPACE`: it specifies how small the volume of a space defined by a pattern will not be considered.
    5. `HUGE_NUMBER`: used as a indication for postive infinity
    5. TODO(now this test is disabled in the code) `MIN_SUPPORT`: a pattern with less than this number of real instances as support will not be considered 



# Utilities for inspection

1. How to inspect the pattern of a node?

    tmp = node.pattern.string_representation(domain=self.domain)
    print(tmp)
2. How to inspect some variables of a node? Use

    print(node.q_value)
    print(node.count)
    ...


# Deprecated:

For compatibility, loading as a scikit-learn `bunch` dictionary object as the dataset is also supported.  Basically, it should have the following attribute :
1. "data":
2. "target": zero or one indicating the class. (multi-class setting is not in the scope of our work)
3. "feature_names": a list of feature name excluding the target label.
4. "target_class": the name of target label, for example "sex"
5. "target_names": the meaning possible class label, for example ["male","female"]. In our approach should be two class be default.
6. "categorical_names" and "categorical_features": a list of be one of "categorical" feature names
7. "class_names": same as "target_names", only for compatibility
8. "class_target": same as "target_class", only for compatibility

Optionally (for compatibility, again) , the dataset also has a dataset (train-validation-test split). Access with `dataset["train"]` and `dataset["labels_train"]`. Replace `train` with `test` or `validation`.
