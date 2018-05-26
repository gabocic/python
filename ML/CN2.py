#!/home/gabriel/pythonenvs/v3.5/bin/python

import Orange
import numpy as np
from time import time
from sklearn.model_selection import train_test_split


def CN2_classifier(data,labels):

    ## Creating an Orange "Domain" to work with the data and the tags
    # Define each attribute
    l_attr = []
    for i in range(0,data.shape[1]):
        l_attr.append(Orange.data.ContinuousVariable(name='f'+i.__str__(), compute_value=None))

    # Define target value
    l_label = [str(i) for i in np.unique(labels).tolist()]
    #l_label = ['a']

    #print('labels',l_label)
    
    classv = Orange.data.DiscreteVariable(name='classv',values=l_label)

    # Create domain based on the above attributes
    mydomain = Orange.data.Domain(attributes=l_attr,class_vars=classv)

    ## Split the dataset into a training and and a testing set
    X_train, X_test, y_train, y_test = train_test_split(data,labels,test_size=0.2,random_state=0)

    ## Loading data and tags in ndarray format into a an Orange.Table
    table_train = Orange.data.Table.from_numpy(mydomain,X_train,Y=y_train)
    table_test = Orange.data.Table.from_numpy(mydomain,X_test,Y=y_test)

    # construct the learning algorithm and use it to induce a classifier
    learner = Orange.classification.CN2Learner()

    # consider up to 10 solution streams at one time
    learner.rule_finder.search_algorithm.beam_width = 10

    # continuous value space is constrained to reduce computation time
    learner.rule_finder.search_strategy.constrain_continuous = True


    ###  Cross-validation
    ###  Test what value of min_samples_leaf produces better results as per AUC
    l_scores=[]
    # min_samples_leaf range: 5% to 14%
    for msl in range(5,15):
        learner.rule_finder.general_validator.min_covered_examples = msl/100
        cv = Orange.evaluation.CrossValidation(table_train, [learner], k=5)
        auc = Orange.evaluation.AUC(cv)
        
        print('msl:',msl,'AUC mean:',auc[0])
        l_scores.append(auc[0])

    winneridx = np.argmax(l_scores)
    winnerpct = winneridx +5
    print('winner %',winnerpct)

    # Instantiate a classifier with the winning min_samples_leaf
    learner.rule_finder.general_validator.min_covered_examples = winnerpct/100

    # Initial time mark
    t0 = time()

    classifier = learner(table_test)

    # Calculate process time
    elap_time = (time() - t0)

    # Obtain predicted labels array
    predicted_labels_prob = classifier.predict()
    predicted_labels = np.argmax(predicted_labels_prob,1)

    # Generate rules dictionary
    l_rules={}
    ruleid = 0
    for myrule in classifier.rule_list:
        l_rules[ruleid]={}
        n_covexamples = 0
        for boolean in myrule.covered_examples:
            if boolean:
                n_covexamples+=1
        l_rules[ruleid]['classes_matched'] = np.zeros((1,max(labels)+1))
        #print(type(myrule.domain.class_var.values[myrule.prediction]))
        l_rules[ruleid]['classes_matched'][0,int(myrule.domain.class_var.values[myrule.prediction])] = n_covexamples
        l_rules[ruleid]['rules'] = []
        for selector in myrule.selectors:
            subrule = {'feature':selector.column,'symbol':selector.op,'threshold':selector.value}
            l_rules[ruleid]['rules'].append(subrule)
        #print('covered examples:',n_covexamples)
        #print('prediction:',myrule.domain.class_var.values[myrule.prediction])
        #print(l_rules[ruleid]['classes_matched'])
        #print(myrule.curr_class_dist)
        #print(dir(myrule.domain))
        #print(myrule.domain.class_var)
        ruleid+=1
    return l_rules,elap_time,predicted_labels,y_test
