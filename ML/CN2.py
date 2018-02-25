#!/home/gabriel/pythonenvs/v3.5/bin/python

import Orange
from sklearn import datasets
import numpy as np



def CN2_classifier(data,estimator):

    ## Creating an Orange "Domain" to work with the data and the tags
    # Define each attribute
    l_attr = []
    for i in range(0,data.shape[1]):
        l_attr.append(Orange.data.ContinuousVariable(name='f'+i.__str__(), compute_value=None))

    # Define target value
    #classv = Orange.data.DiscreteVariable(name='classv',values=estimator.labels_)
    #print(estimator.labels_.tolist())
    
    l_label = [str(i) for i in np.unique(estimator.labels_).tolist()]
    
    #print('<<<<<<<<<<<<<<<<<<<<< EN CN2!! <<<<',estimator.labels_)
    #for label in estimator.labels_:
    #    if label != -1:
    #        l_label.append(label.__str__())    
    classv = Orange.data.DiscreteVariable(name='classv',values=l_label)

    # Create domain based on the above attributes
    mydomain = Orange.data.Domain(attributes=l_attr,class_vars=classv)

    ## Loading data and tags in ndarray format into a an Orange.Table
    table = Orange.data.Table.from_numpy(mydomain,data,Y=estimator.labels_)

    # construct the learning algorithm and use it to induce a classifier
    learner = Orange.classification.CN2Learner()

    # consider up to 10 solution streams at one time
    learner.rule_finder.search_algorithm.beam_width = 10

    # continuous value space is constrained to reduce computation time
    learner.rule_finder.search_strategy.constrain_continuous = True

    # found rules must cover at least 15 examples
    learner.rule_finder.general_validator.min_covered_examples = data.shape[0]*0.1

    # found rules may combine at most 2 selectors (conditions)
    #learner.rule_finder.general_validator.max_rule_length = 3

    classifier = learner(table)

    # Generate rules dictionary
    l_rules={}
    ruleid = 0
    for myrule in classifier.rule_list:
        #print(myrule.initial_class_dist)
        #print(myrule.prior_class_dist)
        l_rules[ruleid]={}
        n_covexamples = 0
        for boolean in myrule.covered_examples:
            if boolean:
                n_covexamples+=1
        l_rules[ruleid]['classes_matched'] = np.zeros((1,max(estimator.labels_)+1))
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
    return l_rules
