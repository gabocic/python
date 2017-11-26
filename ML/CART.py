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

    ## Rules extractor

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

	# If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    l_rules=[]
    n_nodes = clf.tree_.node_count
    for i in range(n_nodes):
        if is_leaves[i]:
            print("Close rule")
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i),clf.tree_.value[i])
        else:
            curr_rule
            print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                    "node %s."
                     % (node_depth[i] * "\t",
		     i,
		     children_left[i],
		     feature[i],
		     threshold[i],
		     children_right[i],
                ))




    #tree_to_code(clf,feature_names)
