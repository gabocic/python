#!/home/gabriel/pythonenvs/v3.5/bin/python

import Orange
import numpy as np
from time import time
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
from threading import Thread
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score


def CN2_classifier(X_train, X_test, y_train,y_test):

    ## Creating an Orange "Domain" to work with the data and the tags
    # Define each attribute
    l_attr = []
    for i in range(0,X_train.shape[1]):
        l_attr.append(Orange.data.ContinuousVariable(name='f'+i.__str__(), compute_value=None))
    
    ulabels = np.unique(np.concatenate((y_train,y_test)))

    # Define target value
    l_label = [str(i) for i in ulabels.tolist()]
    #l_label = ['a']

    #print('labels',l_label)
    
    classv = Orange.data.DiscreteVariable(name='classv',values=l_label)

    # Create domain based on the above attributes
    mydomain = Orange.data.Domain(attributes=l_attr,class_vars=classv)

    # construct the learning algorithm and use it to induce a classifier
    #learner = Orange.classification.CN2Learner()

    # consider up to 10 solution streams at one time
    #learner.rule_finder.search_algorithm.beam_width = 10

    # continuous value space is constrained to reduce computation time
    #learner.rule_finder.search_strategy.constrain_continuous = True

    ## Split the dataset into a training and and a testing set
    #X_train, X_test, y_train, y_test = train_test_split(data,labels,test_size=0.2,random_state=0)

    # Obtain unique labels
    print('ulabels:', ulabels)

    # Binarize labels
    #y_train_bin = label_binarize(y_train, classes=ulabels)
    #n_classes = y_train_bin.shape[1]


    def threadFitData(kf_train_index,kf_test_index,results,index,msl):
        learner = Orange.classification.CN2Learner()
        learner.rule_finder.search_algorithm.beam_width = 10
        learner.rule_finder.search_strategy.constrain_continuous = True

        kf_X_train, kf_X_test = X_train[kf_train_index], X_train[kf_test_index]
        kf_y_train, kf_y_test = y_train[kf_train_index], y_train[kf_test_index]

        learner.rule_finder.general_validator.min_covered_examples = (msl/100)*kf_X_train.shape[0]
        #learner.rule_finder.general_validator.min_covered_examples = 0.15*data.shape[0]

        ## Loading data and tags in ndarray format into a an Orange.Table
        table_train = Orange.data.Table.from_numpy(mydomain,kf_X_train,Y=kf_y_train)
        #table_test = Orange.data.Table.from_numpy(mydomain,kf_X_test,Y=kf_y_test)

        # Fit training set
        classifier = learner(table_train)

        # Obtain predicted labels array
        predicted_labels_prob = classifier.predict(kf_X_test)
        predicted_labels = np.argmax(predicted_labels_prob,1)

        # Binarize labels

        # A single column array is generated when we have two classes only. That's why we need the following workaround
        if len(ulabels) == 2:
            kf_y_test_bin = np.array([[1,0] if l==0 else [0,1] for l in kf_y_test])
        else:
            kf_y_test_bin = label_binarize(kf_y_test, classes=ulabels)
        n_classes = kf_y_test_bin.shape[1]

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

        results[index] = roc_auc["micro"]


    ###  Cross-validation
    ###  Test what value of min_samples_leaf produces better results as per AUC
    l_scores=[]
    splits=5
    # min_samples_leaf range: 5% to 10%
    for msl in range(5,11):
        #learner.rule_finder.general_validator.min_covered_examples = msl/100
        kf = KFold(n_splits=splits)
        #sumauc=0
        index = 0
        threads = [None] * 5
        results = [None] * 5
        for kf_train_index, kf_test_index in kf.split(X_train):
            #threads[index] = Thread(target=threadFitData,args=(kf_train_index,kf_test_index,results,index,msl))
            #threads[index].start()
            threadFitData(kf_train_index,kf_test_index,results,index,msl)
            index+=1




            #kf_X_train, kf_X_test = X_train[kf_train_index], X_train[kf_test_index]
            #kf_y_train, kf_y_test = y_train[kf_train_index], y_train[kf_test_index]

            ### Loading data and tags in ndarray format into a an Orange.Table
            #table_train = Orange.data.Table.from_numpy(mydomain,kf_X_train,Y=kf_y_train)
            ##table_test = Orange.data.Table.from_numpy(mydomain,kf_X_test,Y=kf_y_test)

            ## Fit training set
            #classifier = learner(table_train)

            ## Obtain predicted labels array
            #predicted_labels_prob = classifier.predict(kf_X_test)
            #predicted_labels = np.argmax(predicted_labels_prob,1)

            # Sort y_test as it is pre-requisite
            #s_kf_y_test = np.sort(kf_y_test)
            #auc = metrics.auc(s_kf_y_test,predicted_labels,False)
            #print('auc',auc)
            #sumauc+=auc



        #for i in range(len(threads)):
        #    print('thread ',i)
        #    threads[i].join()

        avg_auc=sum(results)/splits
        print('avg_auc',avg_auc)
        l_scores.append(avg_auc)


        #table_train = Orange.data.Table.from_numpy(mydomain,X_train,Y=y_train)
        #cv = Orange.evaluation.CrossValidation(table_train, [learner], k=5, n_jobs=4)
        #auc = Orange.evaluation.AUC(cv)
        
        #print('msl:',msl,'AUC mean:',auc[0])
        #l_scores.append(auc[0])

    winneridx = np.argmax(l_scores)
    winnerpct = winneridx +5
    print('winner %',winnerpct)
    
    # Instantiate a classifier with the winning min_samples_leaf
    learner = Orange.classification.CN2Learner()
    learner.rule_finder.search_algorithm.beam_width = 10
    learner.rule_finder.search_strategy.constrain_continuous = True
    learner.rule_finder.general_validator.min_covered_examples = (winnerpct/100)*X_train.shape[0]
    table_train = Orange.data.Table.from_numpy(mydomain,X_train,Y=y_train)

    # Initial time mark
    t0 = time()

    classifier = learner(table_train)

    # Calculate process time
    elap_time = (time() - t0)

    # Obtain predicted labels array
    predicted_labels_prob = classifier.predict(X_test)
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
        l_rules[ruleid]['classes_matched'] = np.zeros((1,max(ulabels)+1))
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
    return l_rules,elap_time,predicted_labels,predicted_labels_prob
