#!/home/gabriel/pythonenvs/v3.5/bin/python

from sklearn import tree
from sklearn.tree import _tree
import numpy as np
from time import time
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


import Orange

def CART_classifier(data,labels):

    def tree_to_code(tree, feature_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

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

    ## Main ##

    ## Split the dataset into a training and and a testing set
    X_train, X_test, y_train, y_test = train_test_split(data,labels,test_size=0.2,random_state=0)
    
    ###  Cross-validation
    ###  Test what value of min_samples_leaf produces better results as per AUC

    # Obtain unique labels
    ulabels = np.unique(labels)
    print('ulabels:', ulabels)

    # Binarize labels
    y_train_bin = label_binarize(y_train, classes=ulabels)
    n_classes = y_train_bin.shape[1]

    splits=5
    l_scores=[]
    kf = KFold(n_splits=splits)
    #kfsplits = kf.split(X_train)

    # min_samples_leaf range: 5% to 11%
    for msl in range(5,11):
        clf = tree.DecisionTreeClassifier(min_samples_leaf=msl/100)
        sumauc=0
        for kf_train_index, kf_test_index in kf.split(X_train):
            kf_X_train, kf_X_test = X_train[kf_train_index], X_train[kf_test_index]
            kf_y_train_bin, kf_y_test_bin = y_train_bin[kf_train_index], y_train_bin[kf_test_index]

            # Fit training set
            clf = clf.fit(kf_X_train, kf_y_train_bin)

            # Obtain predicted labels array
            predicted_labels = clf.predict(kf_X_test)
            predicted_labels_prob = clf.predict(kf_X_test)

            # Sort y_test as it is pre-requisite
            #s_kf_y_test = np.sort(kf_y_test)
            #auc = metrics.auc(s_kf_y_test,predicted_labels,False)

            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            #for i in range(n_classes):
            #    fpr[i], tpr[i], _ = roc_curve(kf_y_test_bin[:, i], predicted_labels_prob[:, i])
            #    roc_auc[i] = auc(fpr[i], tpr[i])

            fpr["micro"], tpr["micro"], _ = roc_curve(kf_y_test_bin.ravel(), predicted_labels_prob.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            print('roc_auc["micro"]',roc_auc["micro"])

            sumauc+=roc_auc["micro"]
        avg_auc=sumauc/splits
        print('avg_auc',avg_auc)

        #scores = cross_val_score(clf, X_train, y_train, cv=5,scoring='v_measure_score')
        #print('msl:',msl,'AUC mean:',scores.mean())
        l_scores.append(avg_auc)

    winneridx = np.argmax(l_scores)
    winnerpct = winneridx +5 
    print('winner %',winnerpct)

    # Instantiate a classifier with the winning min_samples_leaf
    clf = tree.DecisionTreeClassifier(min_samples_leaf=winnerpct/100)

    # Initial time mark
    t0 = time()

    clf = clf.fit(X_train, y_train)

    # Calculate process time
    elap_time = (time() - t0)

    # Return predicted label for the testing set 
    predicted_labels = clf.predict(X_test)
    predicted_labels_prob = clf.predict_proba(X_test)

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
#    for i in range(n_nodes):
#        if is_leaves[i]:
#            print("%snode=%s leaf node." % (node_depth[i] * "\t", i),clf.tree_.value[i])
#        else:
#            #curr_rule
#            print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
#                    "node %s."
#                     % (node_depth[i] * "\t",
#    		     i,
#    		     children_left[i],
#    		     feature[i],
#    		     threshold[i],
#    		     children_right[i],
#               ))

    #tree_to_code(clf,feature_names)
    #for regla in l_rules:   
    #    print(l_rules[regla]['classes_matched'])
    return l_rules,elap_time,predicted_labels,y_test,predicted_labels_prob
