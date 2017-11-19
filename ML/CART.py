#!/home/gabriel/pythonenvs/v3.5/bin/python

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import _tree
import numpy as np

def CART_classifier(data,estimator):

    def tree_to_code(tree, feature_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        #print("def tree({}):".format(", ".join(feature_names)))

        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                print("{}if {} <= {}:".format(indent, name, threshold))
                recurse(tree_.children_left[node], depth + 1)
                print("{}else:  # if {} > {}".format(indent, name, threshold))
                recurse(tree_.children_right[node], depth + 1)
            else:
                print("{}return {}".format(indent, tree_.value[node]))

        recurse(0, 1)

    ## Main ##
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(data, estimator.labels_)
    
    # feature_names
    feature_names = np.array([])
    for i in range(0,data.shape[1]):
        feature_names = np.append(feature_names,'f'+i.__str__())

    print(feature_names)

    tree_to_code(clf,feature_names)
