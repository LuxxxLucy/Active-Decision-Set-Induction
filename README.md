# ACTIVE Decision Set Induction from Machine Learning Model
This contains the code of "An Active Approach for Model Interpretation". NeurIPS 2019 Workshop on Human centric machine learning. https://arxiv.org/abs/1910.12207

Note that, the code are used for reproduce the experiments and is not organized very well.

Basically, this is a repo for __decision set__ induction from machine learning model.
This is a model-agnostic approach that could be used to infer human-readable patterns from any blackbox model (classifier).

By human-readable patterns, I mean If-THEN rules.

For reproducibility and open science, competition methods (baselines) are re-implemented or adopted and are included in this repo. It should be easy to reproduce everything.

## Requirements
I assume you use anaconda.
* `python3.6.3
`: all the code and scripts should work on python-3.5. And by a large chance, it would run all fine in python 3.6 and above.
* `scipy`, `numpy`, `scikit-learn>= v0.20`: the packages you should have.
* `Orange`: (I hate pandas!) The Orange package has a good representation of data, especially the `Domain` object.
* `Orange Associate`: package for frequent itemset mining (FP-growth algorithm). `pip install Orange3-Associate`
---
Optionally if you want to reproduce and compare methods, you will needs to install extra packages.
For compared methods (baselines) : in order to reproduce the experiments, you need to install these
* __lime__: this is the dependency for one baseline method (the Anchor algorithm). run `conda install -c conda-forge lime`
For handling text data:
* __spacy__: this is used for processing text data.  
* __pysbrl__ and __pyfim__: for scalable Bayesian rule list.

## Input
---
There is a simple tutorial in `2-d.ipynb` that demonstrate how to prepare the input in the right format in a simple setting: a 2-d dataset and a blackbox function.
In particular, you will need to set these inputs right:
* a classifier model: the blackbox model should be a scikit-learn classifier. And you pass into the algorithm with its `predict` function. For example if `c` is a random forest classifier, you feed as input `c.predict` (note that the data encoder should also be included in this predict function such that the blackbox predict function could operate on raw data.)
* the dataset: for tabular data, the Orange Table is the data structure we'll be working on. It has several very nice properties compared with Pandas dataframe or scikit-learn dictionary-like 'bunch' object. And that is the concept of "domain". Basically, the types of features ("categorical", etc) are explicitly defined in the Orange data table. It avoids the tedious and strange use of scikit "bunch" object.

## Some notes for the competition methods.
---
Most of these baselines are adopted from the original code. A lot of efforts are made to make it work in a unified API and some bugs are fixed.

Passive approaches:

1. __Interpretable decision set__ (this method is rubbish, cannot be reproduced)
3. __Bayesian decision set__ in `BRS.py`: adopted from the original code. fixes some bugs and compatibility issues.
4. __Scalable Bayesian Decision List__. install `pysbrl`
5. __RuleMatrix__ (sampling+SBDL)
6. __CN2__ and __CN2SD__ in `cn2.py`: the famous CN2 sequential-covering algorithm. The CN2SD is a very useful variant (CN2 for subgroup discovery).

Active approaches (bottom-up)
1. __Submodular Pick Anchor__ algorithm in `sp_anchor.py`: first generate a local explanations for every instance, and then greedily pick a good local explanation sequentially for a wide coverage of instances. The original anchor implementation is adopted, which employs a top-down search(general-to-specific search with KL-LUCB algorithm). The code from the original author is only slightly modified, about the data preprocessing. (None of the core code is modified). __We remove the data encoder__ in the original code since an encoder, in our view, is part of the blackbox. We assume the blackbox takes the raw data.
1. __CN2-Anchor__ in `cn2anchor.py`: a simple improve using separate-and-conquer scheme: pick a random instance, generate an anchor, remove the covered instances, and repeat till all instances are covered. We called it CN2-Anchor algorithm. It differs from the above Submodular pick on how to select the explanation: it does pick the best but only randomly. It takes much less time since generating a local explanation is not cheap and it does not have to compute every anchor. But the performance typically is not that different.

Active approaches (top-down):
1. __TREPAN__: Craven 1996. An rather old approach
2. __DTExtract__: See Bastani et al. <Interpreting Blackbox Models via Model Extraction>
3. __UCT-DTExtract__ in `uct.py`: my improvised version, improving DT-extract

## Hyper-Parameters

There are hyper-parameters for this algorithm:

## Notebooks

(Note: the notebooks are not included in the git repo for now)
* `sanity-check-dataset-loading-and-one-instance-explanation.ipynb`: a sanity check
* `2d-experiment.ipynb`: experiments on 2d sinusoidal data and visualization.

## Structure of Code

## Utilities for inspection

1. How to inspect the pattern of a node?

2. How to inspect some variables of a node? Use

## More on implementation limitation

* the implementation of k-nearest neighbor, as you might know, is hard and slow.
