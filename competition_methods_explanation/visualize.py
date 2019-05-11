import numpy as np
import os
import string
from io import open
import json

from .utils import discretizer_from_dataset


class Explanation_Visualizer:
    """
    object for visualize the expalnation
    """
    def __init__(self, type_, exp_map, dataset, discretizer=None):
        self.type = type_
        self.exp_map = exp_map
        self.class_names = dataset.class_names
        self.categorical_features = sorted(dataset.categorical_names.keys())
        self.categorical_names = dataset.categorical_names
        self.feature_names = dataset.feature_names
        if discretizer:
            print("use preivous discretizer")
            self.disc = discretizer
        else:
            self.disc = discretizer_from_dataset(dataset)

    def __init__(self, explanation, dataset, discretizer=None):
        self.type = explanation.type
        self.exp_map = explanation.exp_map
        self.class_names = dataset.class_names
        self.categorical_features = sorted(dataset.categorical_names.keys())
        self.categorical_names = dataset.categorical_names
        self.feature_names = dataset.feature_names
        if discretizer:
            print("use preivous discretizer")
            self.disc = discretizer
        else:
            self.disc = discretizer_from_dataset(dataset)

    def visualize_in_html(self, **kwargs):
        from IPython.core.display import display, HTML
        out = self.as_html(self.exp_map,**kwargs)
        display(HTML(out))
        return


    def transform_to_examples(self, examples, features_in_anchor=[],
                              predicted_label=None):
        ret_obj = []
        if len(examples) == 0:
            return ret_obj
        weights = [int(predicted_label) if x in features_in_anchor else -1
                   for x in range(examples.shape[1])]
        examples = self.disc.discretize(examples)
        for ex in examples:
            values = [self.categorical_names[i][int(ex[i])]
                      if i in self.categorical_features
                      else ex[i] for i in range(ex.shape[0])]
            ret_obj.append(list(zip(self.feature_names, values, weights)))
        return ret_obj

    def to_explanation_map(self, exp):
        def jsonize(x): return json.dumps(x)
        instance = exp['instance']
        predicted_label = exp['prediction']
        predict_proba = np.zeros(len(self.class_names))
        predict_proba[predicted_label] = 1

        examples_obj = []
        for i, temp in enumerate(exp['examples'], start=1):
            features_in_anchor = set(exp['feature'][:i])
            ret = {}
            ret['coveredFalse'] = self.transform_to_examples(
                temp['covered_false'], features_in_anchor, predicted_label)
            ret['coveredTrue'] = self.transform_to_examples(
                temp['covered_true'], features_in_anchor, predicted_label)
            ret['uncoveredTrue'] = self.transform_to_examples(
                temp['uncovered_true'], features_in_anchor, predicted_label)
            ret['uncoveredFalse'] = self.transform_to_examples(
                temp['uncovered_false'], features_in_anchor, predicted_label)
            ret['covered'] =self.transform_to_examples(
                temp['covered'], features_in_anchor, predicted_label)
            examples_obj.append(ret)

        explanation = {'names': exp['names'],
                       'certainties': exp['precision'] if len(exp['precision']) else [exp['all_precision']],
                       'supports': exp['coverage'],
                       'allPrecision': exp['all_precision'],
                       'examples': examples_obj,
                       'onlyShowActive': False}
        weights = [-1 for x in range(instance.shape[0])]
        instance = self.disc.discretize(exp['instance'].reshape(1, -1))[0]
        values = [self.categorical_names[i][int(instance[i])]
                  if i in self.categorical_features
                  else instance[i] for i in range(instance.shape[0])]
        raw_data = list(zip(self.feature_names, values, weights))
        ret = {
            'explanation': explanation,
            'rawData': raw_data,
            'predictProba': list(predict_proba),
            'labelNames': list(map(str, self.class_names)),
            'rawDataType': 'tabular',
            'explanationType': 'anchor',
            'trueClass': False
        }
        return ret

    def as_html(self, exp, **kwargs):
        """bla"""

        def id_generator(size=15):
            """Helper function to generate random div ids. This is useful for embedding
            HTML into ipython notebooks."""
            chars = list(string.ascii_uppercase + string.digits)
            return ''.join(np.random.choice(chars, size, replace=True))

        exp_map = self.to_explanation_map(exp)

        def jsonize(x): return json.dumps(x)
        this_dir, _ = os.path.split(__file__)
        bundle = open(os.path.join(this_dir, 'bundle.js'), encoding='utf8').read()
        random_id = 'top_div' + id_generator()
        out = u'''<html>
        <meta http-equiv="content-type" content="text/html; charset=UTF8">
        <head><script>%s </script></head><body>''' % bundle
        out += u'''
        <div id="{random_id}" />
        <script>
            div = d3.select("#{random_id}");
            lime.RenderExplanationFrame(div,{label_names}, {predict_proba},
            {true_class}, {explanation}, {raw_data}, "tabular", {explanation_type});
        </script>'''.format(random_id=random_id,
                            label_names=jsonize(exp_map['labelNames']),
                            predict_proba=jsonize(exp_map['predictProba']),
                            true_class=jsonize(exp_map['trueClass']),
                            explanation=jsonize(exp_map['explanation']),
                            raw_data=jsonize(exp_map['rawData']),
                            explanation_type=jsonize(exp_map['explanationType']))
        out += u'</body></html>'
        return out

    def save_to_file(self, file_path, **kwargs):
        out = self.as_html(**kwargs)
        io.open(file_path, 'w').write(out)
