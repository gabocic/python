#!/home/gabriel/pythonenvs/v3.5/bin/python

import Orange
from sklearn import datasets

# Read some data

## LOADING DATASET FROM FILE
## *************************
## dataspmat is a scipy matrix storing all the data points. 
## and tags is a numpy.ndarray storing tags for each data point
dataspmat,tags = datasets.load_svmlight_file('dataset.svl')

## Converting from SciPy Sparse matrix to numpy ndarray
data = dataspmat.toarray()

## Creating an Orange "Domain" to work with the data and the tags
# Define each attribute
att1 = Orange.data.ContinuousVariable(name='att1', compute_value=None)
att2 = Orange.data.ContinuousVariable(name='att2', compute_value=None)
att3 = Orange.data.ContinuousVariable(name='att3', compute_value=None)
#classv = Orange.data.DiscreteVariable(name='classv', compute_value=None)
classv = Orange.data.StringVariable(name='classv', compute_value=None)

# Create domain based on the above attributes
mydomain = Orange.data.Domain(attributes=[att1,att2,att3],class_vars=classv)

charar = np.chararray((100, 1))

for idx,etiq in np.ndenumerate(tags):
	if etiq == 0:
		charar[idx]='A'
	elif etiq == 1:
		charar[idx]='B'
	elif etiq == 2:
		charar[idx]='C'


## Loading data and tags in ndarray format into a an Orange.Table
table = Orange.data.Table.from_numpy(mydomain,data,Y=charar)

# construct the learning algorithm and use it to induce a classifier

learner = Orange.classification.CN2Learner()

# consider up to 10 solution streams at one time
learner.rule_finder.search_algorithm.beam_width = 10

# continuous value space is constrained to reduce computation time
learner.rule_finder.search_strategy.constrain_continuous = True

# found rules must cover at least 15 examples
learner.rule_finder.general_validator.min_covered_examples = 15

# found rules may combine at most 2 selectors (conditions)
learner.rule_finder.general_validator.max_rule_length = 2

classifier = learner(data)

for myrule in classifier.rule_list:
	print(myrule.__str__())

