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

        #recurse(0, 1)

    ## Main ##
    clf = tree.DecisionTreeClassifier(min_samples_leaf=0.1)
    clf = clf.fit(data, estimator.labels_)
   
    # feature_names
    feature_names = np.array([])
    for i in range(0,data.shape[1]):
        feature_names = np.append(feature_names,'f'+i.__str__())

    ## Rules extractor
            ## Strategy:
            # 1) Search for a leaf node
            # 2) Check children_left and children_right to find it's parent
            # 3) Extract the condition for that parent using 'feature' and 'threshold'
            # 4) Repeat the process until node 0 is reached

    ## http://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html

    def retrieve_parent_rule(i):
        right=0
        parentid = np.argwhere(children_left == i)
        if parentid.size == 0:
            right=1
            parentid = np.argwhere(children_right == i)[0][0]
        else:
            parentid = parentid[0][0]
        if right == 1:
            symbol = '>'
        else:
            symbol = '<='
        rule = {'feature':feature[parentid],'symbol':symbol,'threshold':threshold[parentid]}
        return parentid,rule

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    # Extract children nodes for each node and leaf nodes
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
            # It's a leaf node
            is_leaves[node_id] = True

    # Extract rules for each leaf node
    l_rules={}
    for i in range(n_nodes):
        if is_leaves[i]:
            rules=[]
            parentid,rule = retrieve_parent_rule(i)
            rules.append(rule)
            while parentid != 0:
                parentid,rule = retrieve_parent_rule(parentid)
                rules.append(rule)
            l_rules[i] = {'rules':rules,'classes_matched':clf.tree_.value[i]}
        
    
    ####### Print validation tree ##################
    #for i in range(n_nodes):
    #    if is_leaves[i]:
    #        print("%snode=%s leaf node." % (node_depth[i] * "\t", i),clf.tree_.value[i])
    #    else:
    #        #curr_rule
    #        print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
    #                "node %s."
    #                 % (node_depth[i] * "\t",
    #		     i,
    #		     children_left[i],
    #		     feature[i],
    #		     threshold[i],
    #		     children_right[i],
    #           ))

    #tree_to_code(clf,feature_names)
    #for regla in l_rules:   
    #    print(l_rules[regla]['classes_matched'])
    return l_rules
